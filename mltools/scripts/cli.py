# Skeleton of a CLI

import click

import mltools


@click.command('mltools')
@click.argument('count', type=int, metavar='N')
def cli(count):
    """Echo a value `N` number of times"""
    for i in range(count):
        click.echo(mltools.has_legs)
