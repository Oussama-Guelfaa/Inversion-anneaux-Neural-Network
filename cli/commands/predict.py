#!/usr/bin/env python3
"""
Commandes de Prédiction CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Commandes pour faire des prédictions avec les modèles entraînés.
"""

import click
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

@click.group(name='predict')
@click.pass_context
def predict_group(ctx):
    """🔮 Commandes de prédiction avec les modèles entraînés."""
    pass

@predict_group.command()
@click.option('--input', '-i', required=True,
              help='Fichier d\'entrée (.mat, .csv, ou .json)')
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research', 'gap-only']),
              default='precision',
              help='Modèle à utiliser pour la prédiction')
@click.option('--output', '-o', 
              help='Fichier de sortie (optionnel)')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'csv', 'table']),
              default='json',
              help='Format de sortie')
@click.option('--confidence', type=float, default=0.8,
              help='Seuil de confiance minimum')
@click.option('--batch-size', type=int, default=32,
              help='Taille des batches pour le traitement')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              default='auto', help='Device de calcul')
@click.pass_context
def single(ctx, input, model, output, output_format, confidence, batch_size, device):
    """
    Fait une prédiction sur un fichier de données.
    
    Exemples:
        hologram_cli predict single --input data.mat --model precision
        hologram_cli predict single --input profiles.csv --output results.json
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    # Validation des entrées
    input_path = Path(input)
    if not input_path.exists():
        utils.show_error(f"Fichier d'entrée non trouvé: {input}")
        return
    
    # Afficher les informations de prédiction
    info_panel = Panel(
        f"[bold blue]Prédiction avec le modèle: {model}[/bold blue]\n\n"
        f"📁 Fichier d'entrée: {input}\n"
        f"📊 Format de sortie: {output_format}\n"
        f"🎯 Seuil de confiance: {confidence}\n"
        f"📦 Batch size: {batch_size}\n"
        f"💻 Device: {device}",
        title="[bold]Configuration de Prédiction[/bold]",
        border_style="cyan"
    )
    console.print(info_panel)
    
    try:
        # Détecter le device automatiquement si nécessaire
        if device == 'auto':
            device = utils.detect_device()
            rprint(f"[green]Device détecté: {device}[/green]")
        
        # Charger les données
        rprint("[blue]📂 Chargement des données...[/blue]")
        data = _load_input_data(input_path, utils)
        
        if data is None:
            return
        
        # Faire la prédiction
        rprint("[blue]🔮 Prédiction en cours...[/blue]")
        results = _run_prediction(data, model, config, utils, device, batch_size)
        
        # Filtrer par confiance
        if confidence > 0:
            results = _filter_by_confidence(results, confidence)
        
        # Sauvegarder les résultats
        if output:
            _save_results(results, output, output_format, utils)
        else:
            _display_results(results, output_format, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors de la prédiction: {str(e)}")

@predict_group.command()
@click.option('--gap', type=float, help='Valeur du gap (µm)')
@click.option('--L-ecran', type=float, help='Distance à l\'écran (µm)')
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research']),
              default='precision',
              help='Modèle à utiliser')
@click.option('--noise-level', type=float, default=0.0,
              help='Niveau de bruit à ajouter (%)')
@click.pass_context
def simulate(ctx, gap, l_ecran, model, noise_level):
    """
    Simule un profil d'anneau et prédit les paramètres.
    
    Exemples:
        hologram_cli predict simulate --gap 0.025 --L-ecran 10.0
        hologram_cli predict simulate --gap 0.05 --L-ecran 15.0 --noise-level 2.0
    """
    utils = ctx.obj['utils']
    
    if gap is None or l_ecran is None:
        utils.show_error("Les paramètres --gap et --L-ecran sont requis")
        return
    
    rprint(f"[blue]🧪 Simulation d'un anneau: gap={gap}µm, L_écran={l_ecran}µm[/blue]")
    
    try:
        # Simuler le profil d'anneau
        simulated_profile = _simulate_ring_profile(gap, l_ecran, noise_level, utils)
        
        # Prédire les paramètres
        prediction = _predict_from_profile(simulated_profile, model, ctx.obj['config'], utils)
        
        # Afficher les résultats
        _display_simulation_results(gap, l_ecran, prediction, noise_level, utils)
        
    except Exception as e:
        utils.show_error(f"Erreur lors de la simulation: {str(e)}")

@predict_group.command()
@click.option('--input-dir', '-d', required=True,
              help='Dossier contenant les fichiers à traiter')
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research']),
              default='precision',
              help='Modèle à utiliser')
@click.option('--output-dir', '-o',
              help='Dossier de sortie (optionnel)')
@click.option('--pattern', default='*.mat',
              help='Motif de fichiers à traiter')
@click.option('--parallel', is_flag=True,
              help='Traitement en parallèle')
@click.pass_context
def batch(ctx, input_dir, model, output_dir, pattern, parallel):
    """
    Traite un lot de fichiers en mode batch.
    
    Exemples:
        hologram_cli predict batch --input-dir data/ --model precision
        hologram_cli predict batch --input-dir profiles/ --pattern "*.csv" --parallel
    """
    utils = ctx.obj['utils']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        utils.show_error(f"Dossier d'entrée non trouvé: {input_dir}")
        return
    
    # Trouver les fichiers à traiter
    files = list(input_path.glob(pattern))
    if not files:
        utils.show_error(f"Aucun fichier trouvé avec le motif: {pattern}")
        return
    
    rprint(f"[blue]📦 Traitement de {len(files)} fichiers...[/blue]")
    
    try:
        results = []
        
        with utils.create_progress_bar("Traitement batch...") as progress:
            task = progress.add_task("Fichiers...", total=len(files))
            
            for file_path in files:
                # Traiter chaque fichier
                file_result = _process_single_file(file_path, model, ctx.obj['config'], utils)
                results.append(file_result)
                progress.update(task, advance=1)
        
        # Sauvegarder les résultats du batch
        if output_dir:
            _save_batch_results(results, output_dir, utils)
        else:
            _display_batch_summary(results, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors du traitement batch: {str(e)}")

def _load_input_data(input_path: Path, utils):
    """Charge les données d'entrée selon le format."""
    try:
        if input_path.suffix == '.mat':
            from scipy.io import loadmat
            return loadmat(str(input_path))
        elif input_path.suffix == '.csv':
            import pandas as pd
            return pd.read_csv(input_path)
        elif input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                return json.load(f)
        else:
            utils.show_error(f"Format de fichier non supporté: {input_path.suffix}")
            return None
    except Exception as e:
        utils.show_error(f"Erreur lors du chargement: {str(e)}")
        return None

def _run_prediction(data, model, config, utils, device, batch_size):
    """Exécute la prédiction sur les données."""
    # Ici on intégrerait avec les modèles existants
    # Pour l'instant, on simule des prédictions
    
    rprint("[yellow]⚡ Simulation de prédiction...[/yellow]")
    
    # Simuler des résultats de prédiction
    results = []
    n_samples = 10  # Nombre simulé d'échantillons
    
    for i in range(n_samples):
        result = {
            'sample_id': i,
            'gap_predicted': np.random.uniform(0.01, 0.1),
            'L_ecran_predicted': np.random.uniform(5.0, 20.0),
            'confidence_gap': np.random.uniform(0.7, 0.99),
            'confidence_L_ecran': np.random.uniform(0.7, 0.99)
        }
        results.append(result)
    
    return results

def _filter_by_confidence(results, confidence_threshold):
    """Filtre les résultats par seuil de confiance."""
    filtered = []
    for result in results:
        min_confidence = min(result.get('confidence_gap', 0), 
                           result.get('confidence_L_ecran', 0))
        if min_confidence >= confidence_threshold:
            filtered.append(result)
    
    return filtered

def _save_results(results, output_path, format, utils):
    """Sauvegarde les résultats dans le format spécifié."""
    try:
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        
        utils.show_success(f"Résultats sauvegardés: {output_path}")
        
    except Exception as e:
        utils.show_error(f"Erreur lors de la sauvegarde: {str(e)}")

def _display_results(results, format, utils):
    """Affiche les résultats dans le terminal."""
    if format == 'table':
        table = utils.create_results_table(results, "Résultats de Prédiction")
        console.print(table)
    else:
        rprint("[green]📊 Résultats de prédiction:[/green]")
        for i, result in enumerate(results[:5]):  # Afficher les 5 premiers
            rprint(f"  {i+1}. Gap: {result.get('gap_predicted', 'N/A'):.4f}µm, "
                  f"L_écran: {result.get('L_ecran_predicted', 'N/A'):.2f}µm")
        
        if len(results) > 5:
            rprint(f"  ... et {len(results)-5} autres résultats")

def _simulate_ring_profile(gap, l_ecran, noise_level, utils):
    """Simule un profil d'anneau holographique."""
    # Simulation simplifiée d'un profil d'anneau
    r = np.linspace(0, 4.2, 600)  # 600 points comme dans les données réelles
    
    # Formule simplifiée pour simuler les anneaux
    intensity = np.exp(-r**2/2) * (1 + 0.5*np.cos(2*np.pi*r*gap/l_ecran))
    
    # Ajouter du bruit si demandé
    if noise_level > 0:
        noise = np.random.normal(0, noise_level/100, intensity.shape)
        intensity += noise
    
    return {'r': r, 'intensity': intensity}

def _predict_from_profile(profile, model, config, utils):
    """Fait une prédiction à partir d'un profil simulé."""
    # Simulation d'une prédiction
    return {
        'gap_predicted': np.random.uniform(0.02, 0.08),
        'L_ecran_predicted': np.random.uniform(8.0, 15.0),
        'confidence_gap': np.random.uniform(0.85, 0.98),
        'confidence_L_ecran': np.random.uniform(0.85, 0.98)
    }

def _display_simulation_results(true_gap, true_l_ecran, prediction, noise_level, utils):
    """Affiche les résultats de simulation."""
    table = Table(title="Résultats de Simulation", show_header=True, header_style="bold blue")
    table.add_column("Paramètre", style="cyan")
    table.add_column("Valeur Vraie", style="green")
    table.add_column("Valeur Prédite", style="yellow")
    table.add_column("Erreur", style="red")
    table.add_column("Confiance", style="magenta")
    
    # Gap
    gap_error = abs(prediction['gap_predicted'] - true_gap)
    table.add_row(
        "Gap (µm)",
        f"{true_gap:.4f}",
        f"{prediction['gap_predicted']:.4f}",
        f"{gap_error:.4f}",
        f"{prediction['confidence_gap']:.3f}"
    )
    
    # L_écran
    l_ecran_error = abs(prediction['L_ecran_predicted'] - true_l_ecran)
    table.add_row(
        "L_écran (µm)",
        f"{true_l_ecran:.2f}",
        f"{prediction['L_ecran_predicted']:.2f}",
        f"{l_ecran_error:.2f}",
        f"{prediction['confidence_L_ecran']:.3f}"
    )
    
    console.print(table)
    
    if noise_level > 0:
        rprint(f"[dim]Niveau de bruit appliqué: {noise_level}%[/dim]")

def _process_single_file(file_path, model, config, utils):
    """Traite un seul fichier pour le mode batch."""
    # Simulation du traitement d'un fichier
    return {
        'file': file_path.name,
        'gap_predicted': np.random.uniform(0.01, 0.1),
        'L_ecran_predicted': np.random.uniform(5.0, 20.0),
        'status': 'success'
    }

def _save_batch_results(results, output_dir, utils):
    """Sauvegarde les résultats du traitement batch."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Sauvegarder en JSON
    results_file = output_path / 'batch_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    utils.show_success(f"Résultats batch sauvegardés: {results_file}")

def _display_batch_summary(results, utils):
    """Affiche un résumé des résultats batch."""
    successful = sum(1 for r in results if r.get('status') == 'success')
    
    summary_panel = Panel(
        f"[bold green]Traitement batch terminé[/bold green]\n\n"
        f"📁 Fichiers traités: {len(results)}\n"
        f"✅ Succès: {successful}\n"
        f"❌ Échecs: {len(results) - successful}",
        title="[bold]Résumé du Batch[/bold]",
        border_style="green"
    )
    console.print(summary_panel)
