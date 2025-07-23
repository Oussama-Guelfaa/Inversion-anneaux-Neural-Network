#!/usr/bin/env python3
"""
Commandes de Configuration CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Commandes pour gérer la configuration du CLI.
"""

import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

@click.group(name='config')
@click.pass_context
def config_group(ctx):
    """⚙️ Commandes de gestion de la configuration."""
    pass

@config_group.command()
@click.pass_context
def show(ctx):
    """
    Affiche la configuration actuelle.
    
    Exemples:
        hologram_cli config show
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    rprint("[blue]⚙️  Configuration actuelle[/blue]")
    
    # Afficher le profil actif
    active_profile = config.get('active_profile', 'production')
    profile_config = config.get_active_profile()
    
    # Tableau de configuration générale
    general_table = Table(title="Configuration Générale", show_header=True, header_style="bold blue")
    general_table.add_column("Paramètre", style="cyan")
    general_table.add_column("Valeur", style="green")
    
    general_table.add_row("Version", config.get('version', 'N/A'))
    general_table.add_row("Profil actif", active_profile)
    general_table.add_row("Fichier de config", config.config_file)
    
    console.print(general_table)
    
    # Tableau du profil actif
    profile_table = Table(title=f"Profil: {active_profile}", show_header=True, header_style="bold green")
    profile_table.add_column("Paramètre", style="cyan")
    profile_table.add_column("Valeur", style="yellow")
    
    for key, value in profile_config.items():
        profile_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(profile_table)
    
    # Configuration UI
    ui_config = config.get('ui', {})
    ui_table = Table(title="Interface Utilisateur", show_header=True, header_style="bold magenta")
    ui_table.add_column("Paramètre", style="cyan")
    ui_table.add_column("Valeur", style="yellow")
    
    for key, value in ui_config.items():
        ui_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(ui_table)

@config_group.command()
@click.option('--profile', '-p', required=True,
              type=click.Choice(['production', 'recherche', 'demo']),
              help='Profil à activer')
@click.pass_context
def set_profile(ctx, profile):
    """
    Change le profil actif.
    
    Exemples:
        hologram_cli config set-profile --profile production
        hologram_cli config set-profile --profile recherche
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    try:
        config.set_active_profile(profile)
        config.save_config()
        
        utils.show_success(f"Profil actif changé vers: {profile}")
        
        # Afficher la nouvelle configuration
        new_profile_config = config.get_active_profile()
        
        profile_panel = Panel(
            f"[bold green]Nouveau profil actif: {profile}[/bold green]\n\n" +
            "\n".join([f"• {k.replace('_', ' ').title()}: {v}" for k, v in new_profile_config.items()]),
            title="[bold]Configuration du Profil[/bold]",
            border_style="green"
        )
        console.print(profile_panel)
        
    except Exception as e:
        utils.show_error(f"Erreur lors du changement de profil: {str(e)}")

@config_group.command()
@click.option('--key', '-k', required=True,
              help='Clé de configuration (notation pointée supportée)')
@click.option('--value', '-v', required=True,
              help='Nouvelle valeur')
@click.option('--type', 'value_type',
              type=click.Choice(['string', 'int', 'float', 'bool']),
              default='string',
              help='Type de la valeur')
@click.pass_context
def set(ctx, key, value, value_type):
    """
    Modifie une valeur de configuration.
    
    Exemples:
        hologram_cli config set --key ui.theme --value dark
        hologram_cli config set --key defaults.train.epochs --value 200 --type int
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    try:
        # Convertir la valeur selon le type
        if value_type == 'int':
            converted_value = int(value)
        elif value_type == 'float':
            converted_value = float(value)
        elif value_type == 'bool':
            converted_value = value.lower() in ['true', '1', 'yes', 'on']
        else:
            converted_value = value
        
        # Définir la nouvelle valeur
        config.set(key, converted_value)
        config.save_config()
        
        utils.show_success(f"Configuration mise à jour: {key} = {converted_value}")
        
    except ValueError as e:
        utils.show_error(f"Erreur de conversion de type: {str(e)}")
    except Exception as e:
        utils.show_error(f"Erreur lors de la modification: {str(e)}")

@config_group.command()
@click.option('--key', '-k', required=True,
              help='Clé de configuration à afficher')
@click.pass_context
def get(ctx, key):
    """
    Affiche une valeur de configuration spécifique.
    
    Exemples:
        hologram_cli config get --key active_profile
        hologram_cli config get --key profiles.production.model_path
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    value = config.get(key)
    
    if value is not None:
        value_panel = Panel(
            f"[bold blue]Clé:[/bold blue] {key}\n"
            f"[bold green]Valeur:[/bold green] {value}\n"
            f"[bold yellow]Type:[/bold yellow] {type(value).__name__}",
            title="[bold]Configuration[/bold]",
            border_style="blue"
        )
        console.print(value_panel)
    else:
        utils.show_error(f"Clé de configuration non trouvée: {key}")

@config_group.command()
@click.pass_context
def list_profiles(ctx):
    """
    Liste tous les profils disponibles.
    
    Exemples:
        hologram_cli config list-profiles
    """
    config = ctx.obj['config']
    
    profiles = config.get('profiles', {})
    active_profile = config.get('active_profile', 'production')
    
    rprint("[blue]📋 Profils disponibles[/blue]")
    
    profiles_table = Table(title="Profils de Configuration", show_header=True, header_style="bold blue")
    profiles_table.add_column("Profil", style="cyan")
    profiles_table.add_column("Modèle", style="green")
    profiles_table.add_column("Données", style="yellow")
    profiles_table.add_column("Précision", style="magenta")
    profiles_table.add_column("Actif", style="red")
    
    for profile_name, profile_config in profiles.items():
        is_active = "✓" if profile_name == active_profile else ""
        
        profiles_table.add_row(
            profile_name,
            profile_config.get('model_path', 'N/A')[:30] + "..." if len(profile_config.get('model_path', '')) > 30 else profile_config.get('model_path', 'N/A'),
            profile_config.get('data_path', 'N/A')[:20] + "..." if len(profile_config.get('data_path', '')) > 20 else profile_config.get('data_path', 'N/A'),
            profile_config.get('precision', 'N/A'),
            is_active
        )
    
    console.print(profiles_table)

@config_group.command()
@click.option('--output', '-o',
              help='Fichier de sortie pour la sauvegarde')
@click.pass_context
def export(ctx, output):
    """
    Exporte la configuration actuelle.
    
    Exemples:
        hologram_cli config export --output my_config.yaml
        hologram_cli config export
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    if not output:
        output = f"cli/exports/config_export_{config.get('active_profile', 'default')}.yaml"
    
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Exporter la configuration
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        utils.show_success(f"Configuration exportée: {output_path}")
        
        # Afficher un résumé de l'export
        export_panel = Panel(
            f"[bold blue]Configuration exportée[/bold blue]\n\n"
            f"📁 Fichier: {output_path}\n"
            f"⚙️  Profil actif: {config.get('active_profile', 'N/A')}\n"
            f"📊 Profils inclus: {len(config.get('profiles', {}))}\n"
            f"🔧 Version: {config.get('version', 'N/A')}",
            title="[bold]Export Réussi[/bold]",
            border_style="green"
        )
        console.print(export_panel)
        
    except Exception as e:
        utils.show_error(f"Erreur lors de l'export: {str(e)}")

@config_group.command()
@click.option('--config-file', '-f', required=True,
              help='Fichier de configuration à importer')
@click.option('--merge', is_flag=True,
              help='Fusionner avec la configuration existante')
@click.pass_context
def import_config(ctx, config_file, merge):
    """
    Importe une configuration depuis un fichier.
    
    Exemples:
        hologram_cli config import --config-file backup.yaml
        hologram_cli config import --config-file new_config.yaml --merge
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    config_path = Path(config_file)
    if not config_path.exists():
        utils.show_error(f"Fichier de configuration non trouvé: {config_file}")
        return
    
    try:
        # Charger la nouvelle configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            new_config = yaml.safe_load(f)
        
        if merge:
            # Fusionner avec la configuration existante
            _merge_configs(config.config, new_config)
            rprint("[yellow]Configuration fusionnée[/yellow]")
        else:
            # Remplacer complètement
            config.config = new_config
            rprint("[yellow]Configuration remplacée[/yellow]")
        
        # Sauvegarder
        config.save_config()
        
        utils.show_success(f"Configuration importée depuis: {config_file}")
        
        # Afficher un résumé
        import_panel = Panel(
            f"[bold blue]Configuration importée[/bold blue]\n\n"
            f"📁 Source: {config_file}\n"
            f"🔄 Mode: {'Fusion' if merge else 'Remplacement'}\n"
            f"⚙️  Nouveau profil actif: {config.get('active_profile', 'N/A')}\n"
            f"📊 Profils disponibles: {len(config.get('profiles', {}))}",
            title="[bold]Import Réussi[/bold]",
            border_style="green"
        )
        console.print(import_panel)
        
    except Exception as e:
        utils.show_error(f"Erreur lors de l'import: {str(e)}")

@config_group.command()
@click.pass_context
def reset(ctx):
    """
    Remet la configuration aux valeurs par défaut.
    
    Exemples:
        hologram_cli config reset
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    # Demander confirmation
    if utils.confirm_action("Êtes-vous sûr de vouloir remettre la configuration aux valeurs par défaut ?"):
        try:
            # Remettre aux valeurs par défaut
            config.config = config._get_default_config()
            config.save_config()
            
            utils.show_success("Configuration remise aux valeurs par défaut")
            
            # Afficher la nouvelle configuration
            reset_panel = Panel(
                f"[bold blue]Configuration réinitialisée[/bold blue]\n\n"
                f"⚙️  Profil actif: {config.get('active_profile', 'production')}\n"
                f"📊 Profils disponibles: {len(config.get('profiles', {}))}\n"
                f"🔧 Version: {config.get('version', '1.0.0')}",
                title="[bold]Reset Réussi[/bold]",
                border_style="green"
            )
            console.print(reset_panel)
            
        except Exception as e:
            utils.show_error(f"Erreur lors de la réinitialisation: {str(e)}")
    else:
        rprint("[yellow]Réinitialisation annulée[/yellow]")

@config_group.command()
@click.pass_context
def validate(ctx):
    """
    Valide la configuration actuelle.
    
    Exemples:
        hologram_cli config validate
    """
    config = ctx.obj['config']
    utils = ctx.obj['utils']
    
    rprint("[blue]🔍 Validation de la configuration...[/blue]")
    
    validation_results = _validate_configuration(config, utils)
    
    # Afficher les résultats de validation
    _display_validation_results(validation_results, utils)

def _merge_configs(base_config, new_config):
    """Fusionne deux configurations de manière récursive."""
    for key, value in new_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value

def _validate_configuration(config, utils):
    """Valide la configuration complète."""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Valider la structure de base
    required_keys = ['version', 'active_profile', 'profiles']
    for key in required_keys:
        if not config.get(key):
            results['valid'] = False
            results['errors'].append(f"Clé manquante: {key}")
        else:
            results['info'].append(f"✓ Clé trouvée: {key}")
    
    # Valider le profil actif
    active_profile = config.get('active_profile')
    profiles = config.get('profiles', {})
    
    if active_profile not in profiles:
        results['valid'] = False
        results['errors'].append(f"Profil actif inexistant: {active_profile}")
    else:
        results['info'].append(f"✓ Profil actif valide: {active_profile}")
    
    # Valider chaque profil
    for profile_name, profile_config in profiles.items():
        required_profile_keys = ['model_path', 'data_path', 'output_path']
        for key in required_profile_keys:
            if key not in profile_config:
                results['warnings'].append(f"Clé manquante dans le profil {profile_name}: {key}")
            else:
                # Vérifier que les chemins existent
                path_value = profile_config[key]
                if key.endswith('_path') and not Path(path_value).exists():
                    results['warnings'].append(f"Chemin inexistant dans {profile_name}.{key}: {path_value}")
                else:
                    results['info'].append(f"✓ Chemin valide {profile_name}.{key}")
    
    return results

def _display_validation_results(results, utils):
    """Affiche les résultats de validation."""
    if results['valid'] and not results['errors']:
        utils.show_success("Configuration valide", 
                          f"{len(results['info'])} vérifications réussies")
    else:
        utils.show_error("Configuration invalide", 
                        f"{len(results['errors'])} erreurs trouvées")
    
    # Afficher les détails
    if results['errors']:
        error_table = Table(title="Erreurs", show_header=True, header_style="bold red")
        error_table.add_column("Erreur", style="red")
        for error in results['errors']:
            error_table.add_row(error)
        console.print(error_table)
    
    if results['warnings']:
        warning_table = Table(title="Avertissements", show_header=True, header_style="bold yellow")
        warning_table.add_column("Avertissement", style="yellow")
        for warning in results['warnings']:
            warning_table.add_row(warning)
        console.print(warning_table)
    
    if results['info']:
        info_table = Table(title="Informations", show_header=True, header_style="bold green")
        info_table.add_column("Information", style="green")
        for info in results['info'][:10]:  # Limiter à 10 pour éviter l'encombrement
            info_table.add_row(info)
        if len(results['info']) > 10:
            info_table.add_row(f"... et {len(results['info']) - 10} autres vérifications réussies")
        console.print(info_table)
