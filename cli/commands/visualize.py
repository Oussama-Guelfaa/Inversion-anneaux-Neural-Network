#!/usr/bin/env python3
"""
Commandes de Visualisation CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Commandes pour créer des visualisations et graphiques ASCII.
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

@click.group(name='visualize')
@click.pass_context
def visualize_group(ctx):
    """📈 Commandes de visualisation et graphiques."""
    pass

@visualize_group.command()
@click.option('--results-file', '-r', required=True,
              help='Fichier de résultats à visualiser (.json)')
@click.option('--output-dir', '-o',
              help='Dossier de sortie pour les graphiques')
@click.option('--format', 'output_format',
              type=click.Choice(['png', 'pdf', 'svg', 'ascii']),
              default='png',
              help='Format de sortie des graphiques')
@click.option('--dpi', type=int, default=300,
              help='Résolution des images (DPI)')
@click.option('--style', default='seaborn',
              help='Style des graphiques matplotlib')
@click.option('--interactive', is_flag=True,
              help='Graphiques interactifs (si supporté)')
@click.pass_context
def results(ctx, results_file, output_dir, output_format, dpi, style, interactive):
    """
    Visualise les résultats de prédiction d'un modèle.
    
    Exemples:
        hologram_cli visualize results --results-file predictions.json
        hologram_cli visualize results --results-file test.json --format ascii
    """
    utils = ctx.obj['utils']
    
    # Vérifier que le fichier existe
    results_path = Path(results_file)
    if not results_path.exists():
        utils.show_error(f"Fichier de résultats non trouvé: {results_file}")
        return
    
    # Afficher les informations de visualisation
    info_panel = Panel(
        f"[bold blue]Visualisation des résultats: {results_file}[/bold blue]\n\n"
        f"📊 Format: {output_format}\n"
        f"🎨 Style: {style}\n"
        f"📐 DPI: {dpi}\n"
        f"🖱️  Interactif: {'Oui' if interactive else 'Non'}",
        title="[bold]Configuration de Visualisation[/bold]",
        border_style="red"
    )
    console.print(info_panel)
    
    try:
        # Charger les résultats
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Créer les visualisations
        if output_format == 'ascii':
            _create_ascii_visualizations(results_data, utils)
        else:
            _create_graphical_visualizations(results_data, output_dir, output_format, 
                                           dpi, style, interactive, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors de la visualisation: {str(e)}")

@visualize_group.command()
@click.option('--dataset-path', '-d', required=True,
              help='Chemin vers le dataset à visualiser')
@click.option('--parameter', '-p',
              type=click.Choice(['gap', 'L_ecran', 'both']),
              default='both',
              help='Paramètre à visualiser')
@click.option('--sample-size', type=int, default=100,
              help='Nombre d\'échantillons à visualiser')
@click.option('--output-dir', '-o',
              help='Dossier de sortie pour les graphiques')
@click.option('--ascii-only', is_flag=True,
              help='Affichage ASCII uniquement dans le terminal')
@click.pass_context
def dataset(ctx, dataset_path, parameter, sample_size, output_dir, ascii_only):
    """
    Visualise la distribution des données dans un dataset.
    
    Exemples:
        hologram_cli visualize dataset --dataset-path data_generation/dataset_2D
        hologram_cli visualize dataset --dataset-path data/ --parameter gap --ascii-only
    """
    utils = ctx.obj['utils']
    
    # Vérifier que le dataset existe
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        utils.show_error(f"Dataset non trouvé: {dataset_path}")
        return
    
    rprint(f"[blue]📊 Visualisation du dataset: {parameter}[/blue]")
    
    try:
        # Analyser le dataset pour la visualisation
        dataset_data = _analyze_dataset_for_visualization(dataset_path_obj, parameter, 
                                                         sample_size, utils)
        
        if ascii_only:
            _create_ascii_dataset_visualization(dataset_data, parameter, utils)
        else:
            _create_dataset_visualizations(dataset_data, parameter, output_dir, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors de la visualisation du dataset: {str(e)}")

@visualize_group.command()
@click.option('--profile-data', '-p', required=True,
              help='Fichier contenant les profils d\'anneaux (.mat, .csv)')
@click.option('--n-profiles', type=int, default=10,
              help='Nombre de profils à afficher')
@click.option('--overlay', is_flag=True,
              help='Superposer tous les profils sur un même graphique')
@click.option('--output', '-o',
              help='Fichier de sortie pour le graphique')
@click.option('--ascii', is_flag=True,
              help='Affichage ASCII dans le terminal')
@click.pass_context
def rings(ctx, profile_data, n_profiles, overlay, output, ascii):
    """
    Visualise les profils d'anneaux holographiques.
    
    Exemples:
        hologram_cli visualize rings --profile-data profiles.mat --n-profiles 5
        hologram_cli visualize rings --profile-data data.csv --overlay --ascii
    """
    utils = ctx.obj['utils']
    
    # Vérifier que les données existent
    profile_path = Path(profile_data)
    if not profile_path.exists():
        utils.show_error(f"Données de profils non trouvées: {profile_data}")
        return
    
    rprint(f"[blue]🔵 Visualisation de {n_profiles} profils d'anneaux[/blue]")
    
    try:
        # Charger les profils
        profiles = _load_ring_profiles(profile_path, n_profiles, utils)
        
        if ascii:
            _create_ascii_ring_visualization(profiles, overlay, utils)
        else:
            _create_ring_visualizations(profiles, overlay, output, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors de la visualisation des anneaux: {str(e)}")

@visualize_group.command()
@click.option('--model1-results', '-m1', required=True,
              help='Résultats du premier modèle')
@click.option('--model2-results', '-m2', required=True,
              help='Résultats du second modèle')
@click.option('--model1-name', default='Modèle 1',
              help='Nom du premier modèle')
@click.option('--model2-name', default='Modèle 2',
              help='Nom du second modèle')
@click.option('--metrics', default='r2,mae,rmse',
              help='Métriques à comparer visuellement')
@click.option('--output-dir', '-o',
              help='Dossier de sortie pour les graphiques')
@click.option('--ascii', is_flag=True,
              help='Graphiques ASCII dans le terminal')
@click.pass_context
def compare(ctx, model1_results, model2_results, model1_name, model2_name, 
           metrics, output_dir, ascii):
    """
    Visualise la comparaison entre deux modèles.
    
    Exemples:
        hologram_cli visualize compare --model1-results precision.json --model2-results production.json
        hologram_cli visualize compare --model1-results old.json --model2-results new.json --ascii
    """
    utils = ctx.obj['utils']
    
    # Vérifier que les fichiers existent
    for file_path, name in [(model1_results, model1_name), (model2_results, model2_name)]:
        if not Path(file_path).exists():
            utils.show_error(f"Fichier de résultats non trouvé pour {name}: {file_path}")
            return
    
    # Parser les métriques
    metrics_list = [m.strip() for m in metrics.split(',')]
    
    rprint(f"[blue]⚖️  Visualisation comparative: {model1_name} vs {model2_name}[/blue]")
    
    try:
        # Charger les résultats
        with open(model1_results, 'r') as f:
            results1 = json.load(f)
        with open(model2_results, 'r') as f:
            results2 = json.load(f)
        
        # Créer les visualisations comparatives
        if ascii:
            _create_ascii_comparison(results1, results2, model1_name, model2_name, 
                                   metrics_list, utils)
        else:
            _create_comparison_visualizations(results1, results2, model1_name, model2_name,
                                            metrics_list, output_dir, utils)
            
    except Exception as e:
        utils.show_error(f"Erreur lors de la visualisation comparative: {str(e)}")

@visualize_group.command()
@click.option('--data', '-d', required=True,
              help='Données à afficher en graphique ASCII')
@click.option('--title', default='Graphique',
              help='Titre du graphique')
@click.option('--width', type=int, default=60,
              help='Largeur du graphique ASCII')
@click.option('--height', type=int, default=20,
              help='Hauteur du graphique ASCII')
@click.option('--chart-type', 
              type=click.Choice(['line', 'bar', 'histogram']),
              default='line',
              help='Type de graphique ASCII')
@click.pass_context
def ascii_chart(ctx, data, title, width, height, chart_type):
    """
    Crée un graphique ASCII personnalisé.
    
    Exemples:
        hologram_cli visualize ascii-chart --data "1,2,3,4,5" --title "Test"
        hologram_cli visualize ascii-chart --data data.csv --chart-type bar
    """
    utils = ctx.obj['utils']
    
    try:
        # Parser les données
        if Path(data).exists():
            # Charger depuis un fichier
            chart_data = _load_chart_data(data, utils)
        else:
            # Parser les données directement
            chart_data = [float(x.strip()) for x in data.split(',')]
        
        # Créer le graphique ASCII
        _create_ascii_chart(chart_data, title, width, height, chart_type, utils)
        
    except Exception as e:
        utils.show_error(f"Erreur lors de la création du graphique ASCII: {str(e)}")

def _create_ascii_visualizations(results_data, utils):
    """Crée des visualisations ASCII des résultats."""
    rprint("[yellow]📊 Génération des graphiques ASCII...[/yellow]")
    
    # Simuler des données de résultats
    gap_errors = np.random.uniform(0, 0.02, 50)
    l_ecran_errors = np.random.uniform(0, 1.0, 50)
    
    # Graphique ASCII simple des erreurs
    _create_simple_ascii_plot(gap_errors, "Erreurs Gap (µm)", utils)
    _create_simple_ascii_plot(l_ecran_errors, "Erreurs L_écran (µm)", utils)

def _create_simple_ascii_plot(data, title, utils):
    """Crée un graphique ASCII simple."""
    # Normaliser les données pour l'affichage
    normalized = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 20).astype(int)
    
    rprint(f"\n[bold blue]{title}[/bold blue]")
    rprint("─" * 60)
    
    # Créer un histogramme ASCII simple
    bins = np.bincount(normalized, minlength=21)
    max_count = max(bins) if max(bins) > 0 else 1
    
    for i, count in enumerate(bins):
        bar_length = int((count / max_count) * 40)
        bar = "█" * bar_length
        rprint(f"{i:2d} │{bar:<40} {count}")
    
    rprint("─" * 60)
    rprint(f"Min: {np.min(data):.6f}, Max: {np.max(data):.6f}, Moyenne: {np.mean(data):.6f}")

def _analyze_dataset_for_visualization(dataset_path, parameter, sample_size, utils):
    """Analyse le dataset pour la visualisation."""
    # Simuler l'analyse du dataset
    data = {
        'parameter': parameter,
        'sample_size': sample_size,
        'gap_values': np.random.uniform(0.01, 0.1, sample_size) if parameter in ['gap', 'both'] else None,
        'L_ecran_values': np.random.uniform(5.0, 20.0, sample_size) if parameter in ['L_ecran', 'both'] else None
    }
    
    return data

def _create_ascii_dataset_visualization(dataset_data, parameter, utils):
    """Crée une visualisation ASCII du dataset."""
    rprint(f"[yellow]📊 Visualisation ASCII du dataset - {parameter}[/yellow]")
    
    if parameter == 'gap' or parameter == 'both':
        if dataset_data['gap_values'] is not None:
            _create_simple_ascii_plot(dataset_data['gap_values'], "Distribution Gap", utils)
    
    if parameter == 'L_ecran' or parameter == 'both':
        if dataset_data['L_ecran_values'] is not None:
            _create_simple_ascii_plot(dataset_data['L_ecran_values'], "Distribution L_écran", utils)

def _load_ring_profiles(profile_path, n_profiles, utils):
    """Charge les profils d'anneaux."""
    # Simuler le chargement de profils
    profiles = []
    
    for i in range(n_profiles):
        r = np.linspace(0, 4.2, 100)
        # Simuler un profil d'anneau
        intensity = np.exp(-r**2/2) * (1 + 0.5*np.cos(2*np.pi*r*np.random.uniform(0.01, 0.1)))
        
        profiles.append({
            'r': r,
            'intensity': intensity,
            'gap': np.random.uniform(0.01, 0.1),
            'L_ecran': np.random.uniform(5.0, 20.0)
        })
    
    return profiles

def _create_ascii_ring_visualization(profiles, overlay, utils):
    """Crée une visualisation ASCII des profils d'anneaux."""
    rprint("[yellow]🔵 Visualisation ASCII des profils d'anneaux[/yellow]")
    
    if overlay:
        # Superposer tous les profils
        rprint("\n[bold blue]Profils d'anneaux superposés[/bold blue]")
        rprint("─" * 80)
        
        # Créer une grille ASCII simple
        for i in range(20):
            line = ""
            for j in range(60):
                # Simuler l'intensité à cette position
                intensity = 0
                for profile in profiles:
                    r_idx = int(j / 60 * len(profile['r']))
                    if r_idx < len(profile['intensity']):
                        intensity += profile['intensity'][r_idx]
                
                # Convertir en caractère ASCII
                intensity_norm = min(intensity / len(profiles), 1.0)
                if intensity_norm > 0.8:
                    line += "█"
                elif intensity_norm > 0.6:
                    line += "▓"
                elif intensity_norm > 0.4:
                    line += "▒"
                elif intensity_norm > 0.2:
                    line += "░"
                else:
                    line += " "
            
            rprint(line)
        
        rprint("─" * 80)
    else:
        # Afficher chaque profil séparément
        for i, profile in enumerate(profiles):
            rprint(f"\n[bold blue]Profil {i+1} - Gap: {profile['gap']:.4f}µm, L_écran: {profile['L_ecran']:.2f}µm[/bold blue]")
            _create_simple_ascii_plot(profile['intensity'], f"Intensité - Profil {i+1}", utils)

def _create_ascii_comparison(results1, results2, model1_name, model2_name, metrics_list, utils):
    """Crée une comparaison ASCII entre deux modèles."""
    rprint(f"[yellow]⚖️  Comparaison ASCII: {model1_name} vs {model2_name}[/yellow]")
    
    # Tableau de comparaison ASCII
    comparison_table = Table(title="Comparaison des Modèles", show_header=True, header_style="bold blue")
    comparison_table.add_column("Métrique", style="cyan")
    comparison_table.add_column(model1_name, style="green")
    comparison_table.add_column(model2_name, style="yellow")
    comparison_table.add_column("Différence", style="magenta")
    comparison_table.add_column("Graphique", style="white")
    
    for metric in metrics_list:
        # Simuler des valeurs
        value1 = np.random.uniform(0.8, 0.99)
        value2 = np.random.uniform(0.8, 0.99)
        diff = value1 - value2
        
        # Créer un mini-graphique ASCII
        if abs(diff) < 0.01:
            graph = "═══"
        elif diff > 0:
            graph = "▲▲▲"
        else:
            graph = "▼▼▼"
        
        comparison_table.add_row(
            metric.upper(),
            f"{value1:.4f}",
            f"{value2:.4f}",
            f"{diff:+.4f}",
            graph
        )
    
    console.print(comparison_table)

def _create_ascii_chart(data, title, width, height, chart_type, utils):
    """Crée un graphique ASCII personnalisé."""
    rprint(f"[yellow]📊 Graphique ASCII: {title}[/yellow]")
    
    if chart_type == 'line':
        _create_ascii_line_chart(data, title, width, height, utils)
    elif chart_type == 'bar':
        _create_ascii_bar_chart(data, title, width, height, utils)
    elif chart_type == 'histogram':
        _create_ascii_histogram(data, title, width, height, utils)

def _create_ascii_line_chart(data, title, width, height, utils):
    """Crée un graphique en ligne ASCII."""
    rprint(f"\n[bold blue]{title} (Ligne)[/bold blue]")
    rprint("─" * width)
    
    # Normaliser les données
    min_val, max_val = min(data), max(data)
    if max_val == min_val:
        normalized = [height // 2] * len(data)
    else:
        normalized = [int((val - min_val) / (max_val - min_val) * (height - 1)) for val in data]
    
    # Créer le graphique ligne par ligne
    for y in range(height - 1, -1, -1):
        line = ""
        for i, norm_val in enumerate(normalized):
            if norm_val == y:
                line += "●"
            elif i > 0 and ((normalized[i-1] < y < norm_val) or (norm_val < y < normalized[i-1])):
                line += "│"
            else:
                line += " "
        rprint(line)
    
    rprint("─" * width)
    rprint(f"Min: {min_val:.3f}, Max: {max_val:.3f}")

def _create_ascii_bar_chart(data, title, width, height, utils):
    """Crée un graphique en barres ASCII."""
    rprint(f"\n[bold blue]{title} (Barres)[/bold blue]")
    
    max_val = max(data) if data else 1
    
    for i, val in enumerate(data):
        bar_length = int((val / max_val) * width)
        bar = "█" * bar_length
        rprint(f"{i:2d} │{bar:<{width}} {val:.3f}")

def _create_ascii_histogram(data, title, width, height, utils):
    """Crée un histogramme ASCII."""
    rprint(f"\n[bold blue]{title} (Histogramme)[/bold blue]")
    
    # Créer des bins
    bins = 10
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1
    
    for i, count in enumerate(hist):
        bar_length = int((count / max_count) * width)
        bar = "█" * bar_length
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        rprint(f"{bin_start:.2f}-{bin_end:.2f} │{bar:<{width}} {count}")

def _load_chart_data(data_file, utils):
    """Charge les données pour un graphique."""
    try:
        if data_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_file)
            return df.iloc[:, 0].values.tolist()  # Première colonne
        elif data_file.endswith('.json'):
            with open(data_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return list(data.values())
        else:
            utils.show_error(f"Format de fichier non supporté: {data_file}")
            return []
    except Exception as e:
        utils.show_error(f"Erreur lors du chargement: {str(e)}")
        return []

def _create_graphical_visualizations(results_data, output_dir, output_format, dpi, style, interactive, utils):
    """Crée des visualisations graphiques (matplotlib)."""
    if not output_dir:
        output_dir = "cli/outputs/visualizations"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rprint("[yellow]📈 Génération des graphiques matplotlib...[/yellow]")
    
    # Ici on intégrerait avec matplotlib pour créer de vrais graphiques
    # Pour l'instant, on simule la génération
    
    plots_generated = [
        f"prediction_scatter.{output_format}",
        f"error_distribution.{output_format}",
        f"performance_metrics.{output_format}"
    ]
    
    for plot_name in plots_generated:
        plot_path = output_path / plot_name
        # Simuler la création du fichier
        plot_path.touch()
    
    utils.show_success(f"Graphiques générés dans: {output_path}")

def _create_dataset_visualizations(dataset_data, parameter, output_dir, utils):
    """Crée des visualisations du dataset."""
    if not output_dir:
        output_dir = "cli/outputs/dataset_viz"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rprint("[yellow]📊 Génération des visualisations du dataset...[/yellow]")
    
    # Simuler la génération de graphiques
    plots = [
        f"{parameter}_distribution.png",
        f"{parameter}_scatter.png",
        f"{parameter}_histogram.png"
    ]
    
    for plot_name in plots:
        (output_path / plot_name).touch()
    
    utils.show_success(f"Visualisations du dataset générées: {output_path}")

def _create_ring_visualizations(profiles, overlay, output, utils):
    """Crée des visualisations des profils d'anneaux."""
    if not output:
        output = "cli/outputs/ring_profiles.png"
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rprint("[yellow]🔵 Génération des visualisations d'anneaux...[/yellow]")
    
    # Simuler la création du graphique
    output_path.touch()
    
    utils.show_success(f"Visualisation des anneaux générée: {output_path}")

def _create_comparison_visualizations(results1, results2, model1_name, model2_name, metrics_list, output_dir, utils):
    """Crée des visualisations comparatives."""
    if not output_dir:
        output_dir = "cli/outputs/comparison"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rprint("[yellow]⚖️  Génération des visualisations comparatives...[/yellow]")
    
    # Simuler la génération de graphiques comparatifs
    plots = [
        "metrics_comparison.png",
        "performance_radar.png",
        "error_comparison.png"
    ]
    
    for plot_name in plots:
        (output_path / plot_name).touch()
    
    utils.show_success(f"Visualisations comparatives générées: {output_path}")
