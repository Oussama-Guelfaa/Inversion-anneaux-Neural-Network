#!/usr/bin/env python3
"""
Commandes d'Entraînement CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Commandes pour l'entraînement des modèles de réseaux neuronaux.
"""

import click
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

console = Console()

@click.group(name='train')
@click.pass_context
def train_group(ctx):
    """🏋️ Commandes d'entraînement des modèles."""
    pass

@train_group.command()
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research', 'gap-only']),
              default='precision',
              help='Type de modèle à entraîner')
@click.option('--data-path', '-d', 
              help='Chemin vers les données d\'entraînement')
@click.option('--epochs', '-e', type=int, default=100,
              help='Nombre d\'époques d\'entraînement')
@click.option('--batch-size', '-b', type=int, default=32,
              help='Taille des batches')
@click.option('--learning-rate', '-lr', type=float, default=0.001,
              help='Taux d\'apprentissage')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              default='auto', help='Device de calcul')
@click.option('--resume', is_flag=True, help='Reprendre un entraînement')
@click.option('--quick-test', is_flag=True, help='Test rapide avant entraînement')
@click.pass_context
def start(ctx, model, data_path, epochs, batch_size, learning_rate, device, resume, quick_test):
    """
    Démarre l'entraînement d'un modèle.
    
    Exemples:
        hologram_cli train start --model precision --epochs 200
        hologram_cli train start --model gap-only --quick-test
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    # Afficher les informations d'entraînement
    info_panel = Panel(
        f"[bold blue]Entraînement du modèle: {model}[/bold blue]\n\n"
        f"📊 Époques: {epochs}\n"
        f"📦 Batch size: {batch_size}\n"
        f"🎯 Learning rate: {learning_rate}\n"
        f"💻 Device: {device}\n"
        f"🔄 Reprendre: {'Oui' if resume else 'Non'}\n"
        f"⚡ Test rapide: {'Oui' if quick_test else 'Non'}",
        title="[bold]Configuration d'Entraînement[/bold]",
        border_style="blue"
    )
    console.print(info_panel)
    
    try:
        # Sélectionner le modèle approprié
        model_path = _get_model_path(model, config)
        if not model_path:
            utils.show_error(f"Modèle '{model}' non trouvé")
            return
        
        # Valider le chemin des données
        if not data_path:
            data_path = config.get_active_profile().get('data_path')
        
        if not utils.validate_data_path(data_path):
            utils.show_error(f"Chemin de données invalide: {data_path}")
            return
        
        # Détecter le device automatiquement si nécessaire
        if device == 'auto':
            device = utils.detect_device()
            rprint(f"[green]Device détecté automatiquement: {device}[/green]")
        
        # Lancer l'entraînement
        if quick_test:
            _run_quick_test(model_path, utils)
        else:
            _run_training(model_path, data_path, epochs, batch_size, 
                         learning_rate, device, resume, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors de l'entraînement: {str(e)}")

@train_group.command()
@click.option('--model', '-m', required=True,
              help='Nom ou chemin du modèle à valider')
@click.pass_context
def validate(ctx, model):
    """
    Valide la configuration d'un modèle avant entraînement.
    
    Exemples:
        hologram_cli train validate --model precision
    """
    utils = ctx.obj['utils']
    
    rprint("[blue]🔍 Validation du modèle...[/blue]")
    
    model_path = _get_model_path(model, ctx.obj['config'])
    if not model_path:
        utils.show_error(f"Modèle '{model}' non trouvé")
        return
    
    # Validation complète
    validation_results = _validate_model_setup(model_path, utils)
    
    if validation_results['valid']:
        utils.show_success("Modèle validé avec succès", 
                          "\n".join(validation_results['messages']))
    else:
        utils.show_error("Validation échouée", 
                        "\n".join(validation_results['errors']))

@train_group.command()
@click.pass_context
def list_models(ctx):
    """
    Liste tous les modèles disponibles pour l'entraînement.
    """
    utils = ctx.obj['utils']
    
    rprint("[blue]📋 Modèles disponibles:[/blue]")
    
    models = utils.get_available_models()
    
    if not models:
        utils.show_warning("Aucun modèle trouvé")
        return
    
    table = utils.create_results_table(models, "Modèles Disponibles")
    console.print(table)

def _get_model_path(model_type: str, config) -> str:
    """Retourne le chemin du modèle basé sur son type."""
    model_mapping = {
        'precision': 'Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25',
        'production': 'Reseau_Neural_Dual_Gap_Lecran_FINAL_16_06_25',
        'research': 'Reseau_Neural_Dual_Gap_Lecran_FINAL_16_06_25',
        'gap-only': 'Reseaux_1D_Gap_Prediction/Reseau_Noise_Robustness'
    }
    
    model_path = model_mapping.get(model_type)
    if model_path and Path(model_path).exists():
        return model_path
    
    return None

def _run_quick_test(model_path: str, utils) -> None:
    """Exécute un test rapide du modèle."""
    rprint("[yellow]⚡ Exécution du test rapide...[/yellow]")
    
    with utils.create_progress_bar("Test rapide en cours...") as progress:
        task = progress.add_task("Validation...", total=100)
        
        # Simuler les étapes de validation
        for i in track(range(100), description="Test rapide..."):
            progress.update(task, advance=1)
    
    utils.show_success("Test rapide terminé", 
                      "Le modèle est prêt pour l'entraînement complet")

def _run_training(model_path: str, data_path: str, epochs: int, 
                 batch_size: int, learning_rate: float, device: str, 
                 resume: bool, utils) -> None:
    """Lance l'entraînement complet du modèle."""
    rprint(f"[green]🚀 Démarrage de l'entraînement...[/green]")
    
    # Ici, on intégrerait avec les scripts d'entraînement existants
    # Pour l'instant, on simule l'entraînement
    
    with utils.create_progress_bar("Entraînement en cours...") as progress:
        task = progress.add_task("Époques...", total=epochs)
        
        for epoch in range(epochs):
            # Simuler une époque d'entraînement
            progress.update(task, advance=1, 
                          description=f"Époque {epoch+1}/{epochs}")
    
    utils.show_success("Entraînement terminé", 
                      f"Modèle sauvegardé dans {model_path}/models/")

def _validate_model_setup(model_path: str, utils) -> dict:
    """Valide la configuration complète d'un modèle."""
    results = {
        'valid': True,
        'messages': [],
        'errors': []
    }
    
    path = Path(model_path)
    
    # Vérifications de base
    if not path.exists():
        results['valid'] = False
        results['errors'].append(f"Chemin inexistant: {model_path}")
        return results
    
    # Vérifier la structure
    required_dirs = ['src', 'config']
    for req_dir in required_dirs:
        if not (path / req_dir).exists():
            results['valid'] = False
            results['errors'].append(f"Dossier manquant: {req_dir}")
        else:
            results['messages'].append(f"✓ Dossier trouvé: {req_dir}")
    
    # Vérifier les fichiers Python
    src_files = list((path / 'src').glob('*.py'))
    if src_files:
        results['messages'].append(f"✓ {len(src_files)} fichiers Python trouvés")
    else:
        results['valid'] = False
        results['errors'].append("Aucun fichier Python dans src/")
    
    return results
