#!/usr/bin/env python3
"""
Formatters CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Classes pour le formatage riche et les graphiques ASCII.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import print as rprint

console = Console()

class RichFormatter:
    """
    Classe pour le formatage riche avec Rich.
    
    Fournit des méthodes pour créer des tableaux, panneaux,
    et autres éléments visuels formatés.
    """
    
    def __init__(self, theme: str = "blue"):
        """
        Initialise le formateur.
        
        Args:
            theme (str): Thème de couleur principal
        """
        self.theme = theme
        self.console = console
    
    def create_header_panel(self, title: str, subtitle: str = None, 
                           border_style: str = None) -> Panel:
        """
        Crée un panneau d'en-tête stylé.
        
        Args:
            title (str): Titre principal
            subtitle (str): Sous-titre optionnel
            border_style (str): Style de bordure
            
        Returns:
            Panel: Panneau formaté
        """
        if not border_style:
            border_style = self.theme
        
        content = f"[bold {self.theme}]{title}[/bold {self.theme}]"
        if subtitle:
            content += f"\n[dim]{subtitle}[/dim]"
        
        return Panel(content, border_style=border_style)
    
    def create_metrics_table(self, metrics: Dict[str, Any], 
                           title: str = "Métriques") -> Table:
        """
        Crée un tableau de métriques formaté.
        
        Args:
            metrics (Dict): Dictionnaire des métriques
            title (str): Titre du tableau
            
        Returns:
            Table: Tableau formaté
        """
        table = Table(title=title, show_header=True, header_style=f"bold {self.theme}")
        table.add_column("Métrique", style="cyan")
        table.add_column("Valeur", style="green")
        table.add_column("Unité", style="dim")
        
        for key, value in metrics.items():
            # Déterminer l'unité basée sur la clé
            unit = self._get_metric_unit(key)
            
            # Formater la valeur
            if isinstance(value, float):
                if 'r2' in key.lower() or 'accuracy' in key.lower():
                    formatted_value = f"{value:.4f} ({value*100:.2f}%)"
                else:
                    formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            
            table.add_row(
                key.replace('_', ' ').title(),
                formatted_value,
                unit
            )
        
        return table
    
    def create_comparison_table(self, data1: Dict, data2: Dict, 
                              label1: str, label2: str,
                              title: str = "Comparaison") -> Table:
        """
        Crée un tableau de comparaison entre deux ensembles de données.
        
        Args:
            data1 (Dict): Premières données
            data2 (Dict): Secondes données
            label1 (str): Label pour les premières données
            label2 (str): Label pour les secondes données
            title (str): Titre du tableau
            
        Returns:
            Table: Tableau de comparaison
        """
        table = Table(title=title, show_header=True, header_style=f"bold {self.theme}")
        table.add_column("Métrique", style="cyan")
        table.add_column(label1, style="green")
        table.add_column(label2, style="yellow")
        table.add_column("Différence", style="magenta")
        
        # Trouver les clés communes
        common_keys = set(data1.keys()) & set(data2.keys())
        
        for key in sorted(common_keys):
            val1 = data1[key]
            val2 = data2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val1 - val2
                diff_str = f"{diff:+.6f}"
                if diff > 0:
                    diff_str = f"[green]{diff_str}[/green]"
                elif diff < 0:
                    diff_str = f"[red]{diff_str}[/red]"
            else:
                diff_str = "N/A"
            
            table.add_row(
                key.replace('_', ' ').title(),
                str(val1),
                str(val2),
                diff_str
            )
        
        return table
    
    def create_status_panel(self, status: str, message: str, 
                          details: str = None) -> Panel:
        """
        Crée un panneau de statut coloré.
        
        Args:
            status (str): Type de statut (success, error, warning, info)
            message (str): Message principal
            details (str): Détails optionnels
            
        Returns:
            Panel: Panneau de statut
        """
        status_config = {
            'success': {'color': 'green', 'icon': '✅', 'title': 'Succès'},
            'error': {'color': 'red', 'icon': '❌', 'title': 'Erreur'},
            'warning': {'color': 'yellow', 'icon': '⚠️', 'title': 'Avertissement'},
            'info': {'color': 'blue', 'icon': 'ℹ️', 'title': 'Information'}
        }
        
        config = status_config.get(status, status_config['info'])
        
        content = f"[{config['color']}]{config['icon']} {message}[/{config['color']}]"
        if details:
            content += f"\n\n[dim]{details}[/dim]"
        
        return Panel(
            content,
            title=f"[bold {config['color']}]{config['title']}[/bold {config['color']}]",
            border_style=config['color']
        )
    
    def create_progress_display(self, current: int, total: int, 
                              description: str = "Progression") -> str:
        """
        Crée un affichage de progression textuel.
        
        Args:
            current (int): Valeur actuelle
            total (int): Valeur totale
            description (str): Description de la tâche
            
        Returns:
            str: Chaîne de progression formatée
        """
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total) if total > 0 else 0
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        return f"{description}: [{bar}] {current}/{total} ({percentage:.1f}%)"
    
    def _get_metric_unit(self, metric_key: str) -> str:
        """Détermine l'unité d'une métrique basée sur sa clé."""
        if 'gap' in metric_key.lower():
            return "µm"
        elif 'l_ecran' in metric_key.lower() or 'lecran' in metric_key.lower():
            return "µm"
        elif 'r2' in metric_key.lower():
            return ""
        elif 'accuracy' in metric_key.lower():
            return "%"
        elif 'time' in metric_key.lower():
            return "s"
        elif 'epoch' in metric_key.lower():
            return ""
        else:
            return ""

class ASCIIGraphs:
    """
    Classe pour créer des graphiques ASCII dans le terminal.
    
    Fournit des méthodes pour créer des graphiques en ligne,
    en barres, histogrammes, et autres visualisations ASCII.
    """
    
    def __init__(self, width: int = 60, height: int = 20):
        """
        Initialise le générateur de graphiques ASCII.
        
        Args:
            width (int): Largeur par défaut des graphiques
            height (int): Hauteur par défaut des graphiques
        """
        self.width = width
        self.height = height
    
    def line_chart(self, data: List[float], title: str = "Graphique en Ligne",
                  labels: List[str] = None, width: int = None, 
                  height: int = None) -> str:
        """
        Crée un graphique en ligne ASCII.
        
        Args:
            data (List[float]): Données à afficher
            title (str): Titre du graphique
            labels (List[str]): Labels pour les points de données
            width (int): Largeur du graphique
            height (int): Hauteur du graphique
            
        Returns:
            str: Graphique ASCII formaté
        """
        w = width or self.width
        h = height or self.height
        
        if not data:
            return f"{title}\n(Aucune donnée à afficher)"
        
        # Normaliser les données
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            normalized = [h // 2] * len(data)
        else:
            normalized = [int((val - min_val) / (max_val - min_val) * (h - 1)) for val in data]
        
        # Créer le graphique
        lines = []
        lines.append(f"📈 {title}")
        lines.append("─" * w)
        
        # Dessiner ligne par ligne (de haut en bas)
        for y in range(h - 1, -1, -1):
            line = ""
            for i, norm_val in enumerate(normalized):
                if i < w:
                    if norm_val == y:
                        line += "●"
                    elif i > 0 and ((normalized[i-1] < y < norm_val) or (norm_val < y < normalized[i-1])):
                        line += "│"
                    else:
                        line += " "
            lines.append(line)
        
        lines.append("─" * w)
        lines.append(f"Min: {min_val:.3f}, Max: {max_val:.3f}, Points: {len(data)}")
        
        return "\n".join(lines)
    
    def bar_chart(self, data: List[float], labels: List[str] = None,
                 title: str = "Graphique en Barres", width: int = None) -> str:
        """
        Crée un graphique en barres ASCII.
        
        Args:
            data (List[float]): Données à afficher
            labels (List[str]): Labels pour les barres
            title (str): Titre du graphique
            width (int): Largeur maximale des barres
            
        Returns:
            str: Graphique en barres ASCII
        """
        w = width or self.width
        
        if not data:
            return f"{title}\n(Aucune donnée à afficher)"
        
        max_val = max(data) if data else 1
        
        lines = []
        lines.append(f"📊 {title}")
        lines.append("─" * (w + 20))
        
        for i, val in enumerate(data):
            bar_length = int((val / max_val) * w) if max_val > 0 else 0
            bar = "█" * bar_length
            
            label = labels[i] if labels and i < len(labels) else f"Item {i+1}"
            label = label[:10]  # Limiter la longueur du label
            
            lines.append(f"{label:>10} │{bar:<{w}} {val:.3f}")
        
        lines.append("─" * (w + 20))
        
        return "\n".join(lines)
    
    def histogram(self, data: List[float], bins: int = 10,
                 title: str = "Histogramme", width: int = None) -> str:
        """
        Crée un histogramme ASCII.
        
        Args:
            data (List[float]): Données à analyser
            bins (int): Nombre de bins
            title (str): Titre de l'histogramme
            width (int): Largeur des barres
            
        Returns:
            str: Histogramme ASCII
        """
        w = width or self.width
        
        if not data:
            return f"{title}\n(Aucune donnée à afficher)"
        
        # Créer l'histogramme
        hist, bin_edges = np.histogram(data, bins=bins)
        max_count = max(hist) if max(hist) > 0 else 1
        
        lines = []
        lines.append(f"📊 {title}")
        lines.append("─" * (w + 25))
        
        for i, count in enumerate(hist):
            bar_length = int((count / max_count) * w)
            bar = "█" * bar_length
            
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            
            lines.append(f"{bin_label:>12} │{bar:<{w}} {count}")
        
        lines.append("─" * (w + 25))
        lines.append(f"Total: {len(data)} échantillons, {bins} bins")
        
        return "\n".join(lines)
    
    def scatter_plot(self, x_data: List[float], y_data: List[float],
                    title: str = "Nuage de Points", width: int = None,
                    height: int = None) -> str:
        """
        Crée un nuage de points ASCII.
        
        Args:
            x_data (List[float]): Données X
            y_data (List[float]): Données Y
            title (str): Titre du graphique
            width (int): Largeur du graphique
            height (int): Hauteur du graphique
            
        Returns:
            str: Nuage de points ASCII
        """
        w = width or self.width
        h = height or self.height
        
        if not x_data or not y_data or len(x_data) != len(y_data):
            return f"{title}\n(Données invalides pour le nuage de points)"
        
        # Normaliser les données
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        
        if x_max == x_min:
            x_norm = [w // 2] * len(x_data)
        else:
            x_norm = [int((x - x_min) / (x_max - x_min) * (w - 1)) for x in x_data]
        
        if y_max == y_min:
            y_norm = [h // 2] * len(y_data)
        else:
            y_norm = [int((y - y_min) / (y_max - y_min) * (h - 1)) for y in y_data]
        
        # Créer la grille
        grid = [[' ' for _ in range(w)] for _ in range(h)]
        
        # Placer les points
        for x, y in zip(x_norm, y_norm):
            if 0 <= x < w and 0 <= y < h:
                grid[h - 1 - y][x] = '●'  # Inverser Y pour affichage correct
        
        # Construire l'affichage
        lines = []
        lines.append(f"📈 {title}")
        lines.append("─" * w)
        
        for row in grid:
            lines.append(''.join(row))
        
        lines.append("─" * w)
        lines.append(f"X: {x_min:.3f} à {x_max:.3f}, Y: {y_min:.3f} à {y_max:.3f}")
        
        return "\n".join(lines)
    
    def progress_bar(self, current: int, total: int, 
                    description: str = "Progression", width: int = None) -> str:
        """
        Crée une barre de progression ASCII.
        
        Args:
            current (int): Valeur actuelle
            total (int): Valeur totale
            description (str): Description de la tâche
            width (int): Largeur de la barre
            
        Returns:
            str: Barre de progression ASCII
        """
        w = width or self.width
        
        percentage = (current / total) * 100 if total > 0 else 0
        filled_length = int(w * current // total) if total > 0 else 0
        
        bar = "█" * filled_length + "░" * (w - filled_length)
        
        return f"{description}: [{bar}] {current}/{total} ({percentage:.1f}%)"
    
    def comparison_chart(self, data1: List[float], data2: List[float],
                        label1: str = "Série 1", label2: str = "Série 2",
                        title: str = "Comparaison", width: int = None) -> str:
        """
        Crée un graphique de comparaison ASCII.
        
        Args:
            data1 (List[float]): Première série de données
            data2 (List[float]): Seconde série de données
            label1 (str): Label pour la première série
            label2 (str): Label pour la seconde série
            title (str): Titre du graphique
            width (int): Largeur du graphique
            
        Returns:
            str: Graphique de comparaison ASCII
        """
        w = width or self.width
        
        if not data1 or not data2:
            return f"{title}\n(Données insuffisantes pour la comparaison)"
        
        # Assurer que les deux séries ont la même longueur
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        lines = []
        lines.append(f"⚖️  {title}")
        lines.append("─" * (w + 30))
        lines.append(f"Légende: {label1} (●), {label2} (○)")
        lines.append("─" * (w + 30))
        
        # Créer le graphique de comparaison
        all_data = data1 + data2
        min_val, max_val = min(all_data), max(all_data)
        
        if max_val == min_val:
            norm1 = [w // 2] * len(data1)
            norm2 = [w // 2] * len(data2)
        else:
            norm1 = [int((val - min_val) / (max_val - min_val) * (w - 1)) for val in data1]
            norm2 = [int((val - min_val) / (max_val - min_val) * (w - 1)) for val in data2]
        
        for i in range(min_len):
            line = [' '] * w
            
            # Placer les points
            if 0 <= norm1[i] < w:
                line[norm1[i]] = '●'
            if 0 <= norm2[i] < w:
                if line[norm2[i]] == '●':
                    line[norm2[i]] = '◉'  # Superposition
                else:
                    line[norm2[i]] = '○'
            
            lines.append(f"Point {i+1:2d} │{''.join(line)}│ {data1[i]:.3f} / {data2[i]:.3f}")
        
        lines.append("─" * (w + 30))
        
        return "\n".join(lines)
