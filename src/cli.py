import click

@click.command()
@click.option('--model', type=click.Path(), help='Path to the model file')
@click.option('--data', type=click.Path(), help='Path to the dataset file')
@click.option('--attack', multiple=True, help='Attack types to run (e.g., fgsm, pgd)')
def main(model, data, attack):
    click.echo(f'Model: {model}')
    click.echo(f'Data: {data}')
    click.echo(f'Attacks selected: {attack}')

if __name__ == '__main__':
    main()
