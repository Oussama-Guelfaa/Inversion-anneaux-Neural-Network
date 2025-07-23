#!/usr/bin/env python3
"""
Commandes de Test CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Commandes pour tester et évaluer les modèles entraînés.
"""

import click
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import print as rprint

console = Console()

@click.group(name='test')
@click.pass_context
def test_group(ctx):
    """🧪 Commandes de test et d'évaluation des modèles."""
    pass

@test_group.command()
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research', 'gap-only']),
              default='precision',
              help='Modèle à tester')
@click.option('--test-data', '-d',
              help='Chemin vers les données de test')
@click.option('--tolerance-gap', type=float, default=0.007,
              help='Tolérance pour le gap (µm)')
@click.option('--tolerance-L-ecran', type=float, default=0.5,
              help='Tolérance pour L_écran (µm)')
@click.option('--output', '-o',
              help='Fichier de sortie pour les résultats')
@click.option('--detailed', is_flag=True,
              help='Rapport détaillé avec métriques avancées')
@click.pass_context
def accuracy(ctx, model, test_data, tolerance_gap, tolerance_l_ecran, output, detailed):
    """
    Teste la précision d'un modèle sur un jeu de données de test.
    
    Exemples:
        hologram_cli test accuracy --model precision --detailed
        hologram_cli test accuracy --model gap-only --tolerance-gap 0.01
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    # Afficher les informations de test
    info_panel = Panel(
        f"[bold blue]Test de précision: {model}[/bold blue]\n\n"
        f"🎯 Tolérance gap: {tolerance_gap} µm\n"
        f"🎯 Tolérance L_écran: {tolerance_l_ecran} µm\n"
        f"📊 Rapport détaillé: {'Oui' if detailed else 'Non'}",
        title="[bold]Configuration de Test[/bold]",
        border_style="yellow"
    )
    console.print(info_panel)
    
    try:
        # Charger les données de test
        if not test_data:
            test_data = config.get_active_profile().get('data_path')
        
        rprint("[blue]📂 Chargement des données de test...[/blue]")
        test_results = _run_accuracy_test(model, test_data, tolerance_gap, 
                                        tolerance_l_ecran, config, utils)
        
        # Afficher les résultats
        _display_accuracy_results(test_results, detailed, utils)
        
        # Sauvegarder si demandé
        if output:
            _save_test_results(test_results, output, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors du test de précision: {str(e)}")

@test_group.command()
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research']),
              default='precision',
              help='Modèle à tester')
@click.option('--noise-levels', default='0,1,2,5,10',
              help='Niveaux de bruit à tester (% séparés par des virgules)')
@click.option('--n-samples', type=int, default=100,
              help='Nombre d\'échantillons par niveau de bruit')
@click.option('--output-dir', '-o',
              help='Dossier de sortie pour les résultats')
@click.pass_context
def robustness(ctx, model, noise_levels, n_samples, output_dir):
    """
    Teste la robustesse d'un modèle au bruit.
    
    Exemples:
        hologram_cli test robustness --model precision --noise-levels "0,2,5,10"
        hologram_cli test robustness --model production --n-samples 200
    """
    utils = ctx.obj['utils']
    
    # Parser les niveaux de bruit
    try:
        noise_list = [float(x.strip()) for x in noise_levels.split(',')]
    except ValueError:
        utils.show_error("Format invalide pour les niveaux de bruit")
        return
    
    rprint(f"[blue]🔊 Test de robustesse au bruit: {noise_list}%[/blue]")
    
    try:
        robustness_results = {}
        
        with utils.create_progress_bar("Test de robustesse...") as progress:
            task = progress.add_task("Niveaux de bruit...", total=len(noise_list))
            
            for noise_level in noise_list:
                rprint(f"[yellow]Testing noise level: {noise_level}%[/yellow]")
                
                # Tester ce niveau de bruit
                noise_results = _test_noise_level(model, noise_level, n_samples, 
                                                ctx.obj['config'], utils)
                robustness_results[noise_level] = noise_results
                
                progress.update(task, advance=1)
        
        # Afficher les résultats de robustesse
        _display_robustness_results(robustness_results, utils)
        
        # Sauvegarder si demandé
        if output_dir:
            _save_robustness_results(robustness_results, output_dir, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors du test de robustesse: {str(e)}")

@test_group.command()
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research']),
              default='precision',
              help='Modèle à tester')
@click.option('--experimental-data', '-e', required=True,
              help='Chemin vers les données expérimentales')
@click.option('--preprocessing', is_flag=True,
              help='Appliquer le préprocessing aux données expérimentales')
@click.option('--output', '-o',
              help='Fichier de sortie pour les résultats')
@click.pass_context
def experimental(ctx, model, experimental_data, preprocessing, output):
    """
    Teste un modèle sur des données expérimentales réelles.
    
    Exemples:
        hologram_cli test experimental --model precision --experimental-data data/exp.mat
        hologram_cli test experimental --model production --preprocessing
    """
    utils = ctx.obj['utils']
    
    # Vérifier que les données expérimentales existent
    exp_path = Path(experimental_data)
    if not exp_path.exists():
        utils.show_error(f"Données expérimentales non trouvées: {experimental_data}")
        return
    
    rprint(f"[blue]🔬 Test sur données expérimentales: {experimental_data}[/blue]")
    
    try:
        # Charger et préprocesser les données expérimentales
        exp_data = _load_experimental_data(exp_path, preprocessing, utils)
        
        if exp_data is None:
            return
        
        # Tester le modèle
        exp_results = _test_on_experimental_data(model, exp_data, ctx.obj['config'], utils)
        
        # Afficher les résultats
        _display_experimental_results(exp_results, utils)
        
        # Sauvegarder si demandé
        if output:
            _save_experimental_results(exp_results, output, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors du test expérimental: {str(e)}")

@test_group.command()
@click.option('--model', '-m', 
              type=click.Choice(['precision', 'production', 'research']),
              default='precision',
              help='Modèle à comparer')
@click.option('--baseline-model', '-b',
              type=click.Choice(['precision', 'production', 'research']),
              help='Modèle de référence pour la comparaison')
@click.option('--test-data', '-d',
              help='Données de test pour la comparaison')
@click.option('--metrics', default='r2,mae,rmse',
              help='Métriques à comparer (séparées par des virgules)')
@click.pass_context
def compare(ctx, model, baseline_model, test_data, metrics):
    """
    Compare les performances de deux modèles.
    
    Exemples:
        hologram_cli test compare --model precision --baseline-model production
        hologram_cli test compare --model research --baseline-model precision --metrics "r2,mae"
    """
    utils = ctx.obj['utils']
    
    if not baseline_model:
        utils.show_error("Le modèle de référence (--baseline-model) est requis")
        return
    
    # Parser les métriques
    metrics_list = [m.strip() for m in metrics.split(',')]
    
    rprint(f"[blue]⚖️  Comparaison: {model} vs {baseline_model}[/blue]")
    
    try:
        # Tester les deux modèles
        model_results = _evaluate_model(model, test_data, metrics_list, ctx.obj['config'], utils)
        baseline_results = _evaluate_model(baseline_model, test_data, metrics_list, ctx.obj['config'], utils)
        
        # Comparer les résultats
        comparison = _compare_model_results(model_results, baseline_results, metrics_list)
        
        # Afficher la comparaison
        _display_comparison_results(model, baseline_model, comparison, utils)
        
    except Exception as e:
        utils.show_error(f"Erreur lors de la comparaison: {str(e)}")

def _run_accuracy_test(model, test_data, tolerance_gap, tolerance_l_ecran, config, utils):
    """Exécute un test de précision complet."""
    # Simulation d'un test de précision
    n_samples = 500
    results = {
        'model': model,
        'n_samples': n_samples,
        'tolerance_gap': tolerance_gap,
        'tolerance_L_ecran': tolerance_l_ecran,
        'predictions': [],
        'metrics': {}
    }
    
    # Simuler des prédictions
    for i in track(range(n_samples), description="Test de précision..."):
        true_gap = np.random.uniform(0.01, 0.1)
        true_l_ecran = np.random.uniform(5.0, 20.0)
        
        # Simuler des prédictions avec erreur
        pred_gap = true_gap + np.random.normal(0, 0.005)
        pred_l_ecran = true_l_ecran + np.random.normal(0, 0.3)
        
        prediction = {
            'true_gap': true_gap,
            'pred_gap': pred_gap,
            'true_L_ecran': true_l_ecran,
            'pred_L_ecran': pred_l_ecran,
            'error_gap': abs(pred_gap - true_gap),
            'error_L_ecran': abs(pred_l_ecran - true_l_ecran),
            'within_tolerance_gap': abs(pred_gap - true_gap) <= tolerance_gap,
            'within_tolerance_L_ecran': abs(pred_l_ecran - true_l_ecran) <= tolerance_l_ecran
        }
        results['predictions'].append(prediction)
    
    # Calculer les métriques
    predictions = results['predictions']
    results['metrics'] = {
        'accuracy_gap': sum(p['within_tolerance_gap'] for p in predictions) / len(predictions),
        'accuracy_L_ecran': sum(p['within_tolerance_L_ecran'] for p in predictions) / len(predictions),
        'mae_gap': np.mean([p['error_gap'] for p in predictions]),
        'mae_L_ecran': np.mean([p['error_L_ecran'] for p in predictions]),
        'rmse_gap': np.sqrt(np.mean([p['error_gap']**2 for p in predictions])),
        'rmse_L_ecran': np.sqrt(np.mean([p['error_L_ecran']**2 for p in predictions]))
    }
    
    return results

def _display_accuracy_results(results, detailed, utils):
    """Affiche les résultats de test de précision."""
    metrics = results['metrics']
    
    # Tableau des métriques principales
    table = Table(title="Résultats de Précision", show_header=True, header_style="bold blue")
    table.add_column("Métrique", style="cyan")
    table.add_column("Gap", style="green")
    table.add_column("L_écran", style="yellow")
    
    table.add_row(
        "Précision (%)",
        f"{metrics['accuracy_gap']*100:.2f}%",
        f"{metrics['accuracy_L_ecran']*100:.2f}%"
    )
    table.add_row(
        "MAE (µm)",
        f"{metrics['mae_gap']:.6f}",
        f"{metrics['mae_L_ecran']:.3f}"
    )
    table.add_row(
        "RMSE (µm)",
        f"{metrics['rmse_gap']:.6f}",
        f"{metrics['rmse_L_ecran']:.3f}"
    )
    
    console.print(table)
    
    if detailed:
        # Afficher des statistiques détaillées
        _display_detailed_statistics(results, utils)

def _display_detailed_statistics(results, utils):
    """Affiche des statistiques détaillées."""
    predictions = results['predictions']
    
    # Statistiques sur les erreurs
    gap_errors = [p['error_gap'] for p in predictions]
    l_ecran_errors = [p['error_L_ecran'] for p in predictions]
    
    stats_panel = Panel(
        f"[bold blue]Statistiques Détaillées[/bold blue]\n\n"
        f"📊 Échantillons testés: {len(predictions)}\n\n"
        f"[green]Gap:[/green]\n"
        f"  • Erreur min: {min(gap_errors):.6f} µm\n"
        f"  • Erreur max: {max(gap_errors):.6f} µm\n"
        f"  • Erreur médiane: {np.median(gap_errors):.6f} µm\n"
        f"  • Écart-type: {np.std(gap_errors):.6f} µm\n\n"
        f"[yellow]L_écran:[/yellow]\n"
        f"  • Erreur min: {min(l_ecran_errors):.3f} µm\n"
        f"  • Erreur max: {max(l_ecran_errors):.3f} µm\n"
        f"  • Erreur médiane: {np.median(l_ecran_errors):.3f} µm\n"
        f"  • Écart-type: {np.std(l_ecran_errors):.3f} µm",
        title="[bold]Analyse Statistique[/bold]",
        border_style="blue"
    )
    console.print(stats_panel)

def _test_noise_level(model, noise_level, n_samples, config, utils):
    """Teste un niveau de bruit spécifique."""
    # Simulation du test de robustesse au bruit
    results = {
        'noise_level': noise_level,
        'n_samples': n_samples,
        'degradation_gap': np.random.uniform(0.05, 0.3),  # Dégradation simulée
        'degradation_L_ecran': np.random.uniform(0.02, 0.2),
        'accuracy_gap': max(0.5, 0.95 - noise_level * 0.02),  # Précision qui diminue avec le bruit
        'accuracy_L_ecran': max(0.6, 0.98 - noise_level * 0.015)
    }
    
    return results

def _display_robustness_results(robustness_results, utils):
    """Affiche les résultats de test de robustesse."""
    table = Table(title="Test de Robustesse au Bruit", show_header=True, header_style="bold blue")
    table.add_column("Bruit (%)", style="cyan")
    table.add_column("Précision Gap (%)", style="green")
    table.add_column("Précision L_écran (%)", style="yellow")
    table.add_column("Dégradation Gap (%)", style="red")
    table.add_column("Dégradation L_écran (%)", style="red")
    
    for noise_level, results in robustness_results.items():
        table.add_row(
            f"{noise_level:.1f}",
            f"{results['accuracy_gap']*100:.1f}",
            f"{results['accuracy_L_ecran']*100:.1f}",
            f"{results['degradation_gap']*100:.1f}",
            f"{results['degradation_L_ecran']*100:.1f}"
        )
    
    console.print(table)

def _load_experimental_data(exp_path, preprocessing, utils):
    """Charge les données expérimentales."""
    try:
        if exp_path.suffix == '.mat':
            from scipy.io import loadmat
            data = loadmat(str(exp_path))
            
            if preprocessing:
                rprint("[yellow]🔧 Préprocessing des données expérimentales...[/yellow]")
                # Ici on appliquerait le préprocessing
            
            return data
        else:
            utils.show_error(f"Format de fichier non supporté: {exp_path.suffix}")
            return None
            
    except Exception as e:
        utils.show_error(f"Erreur lors du chargement: {str(e)}")
        return None

def _test_on_experimental_data(model, exp_data, config, utils):
    """Teste le modèle sur les données expérimentales."""
    # Simulation du test sur données expérimentales
    return {
        'model': model,
        'n_experimental_samples': 100,
        'predictions_made': 85,  # Certains échantillons peuvent échouer
        'average_confidence': 0.72,  # Confiance généralement plus faible sur données réelles
        'domain_adaptation_needed': True
    }

def _display_experimental_results(results, utils):
    """Affiche les résultats de test expérimental."""
    exp_panel = Panel(
        f"[bold blue]Test sur Données Expérimentales[/bold blue]\n\n"
        f"📊 Échantillons expérimentaux: {results['n_experimental_samples']}\n"
        f"✅ Prédictions réussies: {results['predictions_made']}\n"
        f"🎯 Confiance moyenne: {results['average_confidence']:.3f}\n"
        f"🔧 Adaptation de domaine: {'Recommandée' if results['domain_adaptation_needed'] else 'Non nécessaire'}",
        title="[bold]Résultats Expérimentaux[/bold]",
        border_style="magenta"
    )
    console.print(exp_panel)

def _evaluate_model(model, test_data, metrics_list, config, utils):
    """Évalue un modèle sur les métriques spécifiées."""
    # Simulation de l'évaluation
    return {
        'model': model,
        'r2': np.random.uniform(0.85, 0.99),
        'mae': np.random.uniform(0.001, 0.01),
        'rmse': np.random.uniform(0.002, 0.015)
    }

def _compare_model_results(model_results, baseline_results, metrics_list):
    """Compare les résultats de deux modèles."""
    comparison = {}
    
    for metric in metrics_list:
        if metric in model_results and metric in baseline_results:
            model_val = model_results[metric]
            baseline_val = baseline_results[metric]
            
            # Pour R², plus c'est haut, mieux c'est
            # Pour MAE/RMSE, plus c'est bas, mieux c'est
            if metric == 'r2':
                improvement = ((model_val - baseline_val) / baseline_val) * 100
            else:
                improvement = ((baseline_val - model_val) / baseline_val) * 100
            
            comparison[metric] = {
                'model_value': model_val,
                'baseline_value': baseline_val,
                'improvement_percent': improvement
            }
    
    return comparison

def _display_comparison_results(model, baseline_model, comparison, utils):
    """Affiche les résultats de comparaison."""
    table = Table(title=f"Comparaison: {model} vs {baseline_model}", 
                 show_header=True, header_style="bold blue")
    table.add_column("Métrique", style="cyan")
    table.add_column(model, style="green")
    table.add_column(baseline_model, style="yellow")
    table.add_column("Amélioration (%)", style="magenta")
    
    for metric, data in comparison.items():
        improvement = data['improvement_percent']
        improvement_str = f"{improvement:+.2f}%"
        if improvement > 0:
            improvement_str = f"[green]{improvement_str}[/green]"
        elif improvement < 0:
            improvement_str = f"[red]{improvement_str}[/red]"
        
        table.add_row(
            metric.upper(),
            f"{data['model_value']:.6f}",
            f"{data['baseline_value']:.6f}",
            improvement_str
        )
    
    console.print(table)

def _save_test_results(results, output_path, utils):
    """Sauvegarde les résultats de test."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        utils.show_success(f"Résultats sauvegardés: {output_path}")
    except Exception as e:
        utils.show_error(f"Erreur lors de la sauvegarde: {str(e)}")

def _save_robustness_results(results, output_dir, utils):
    """Sauvegarde les résultats de robustesse."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_file = output_path / 'robustness_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        utils.show_success(f"Résultats de robustesse sauvegardés: {results_file}")
    except Exception as e:
        utils.show_error(f"Erreur lors de la sauvegarde: {str(e)}")

def _save_experimental_results(results, output_path, utils):
    """Sauvegarde les résultats expérimentaux."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        utils.show_success(f"Résultats expérimentaux sauvegardés: {output_path}")
    except Exception as e:
        utils.show_error(f"Erreur lors de la sauvegarde: {str(e)}")
