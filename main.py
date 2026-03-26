"""
Entry point for the Mood Machine rule based mood analyzer.
"""

from typing import List

from rich.console import Console
from rich.table import Table
from rich import print

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS

console = Console()


def evaluate_rule_based(posts: List[str], labels: List[str]) -> float:
    """
    Evaluate the rule based MoodAnalyzer on a labeled dataset.

    Prints each text with its predicted label and the true label,
    then returns the overall accuracy as a float between 0 and 1.
    """
    analyzer = MoodAnalyzer()
    correct = 0
    total = len(posts)

    table = Table(title="Rule Based Evaluation")

    table.add_column("Text", style="cyan", no_wrap=False)
    table.add_column("Predicted", style="magenta")
    table.add_column("Actual", style="green")
    table.add_column("Correct?", justify="center")

    for text, true_label in zip(posts, labels):
        predicted_label = analyzer.predict_label(text)
        is_correct = predicted_label == true_label
        
        if is_correct:
            correct += 1
            correct_mark = "[green]✔[/green]"
        else:
            correct_mark = "[red]✘[/red]"

        table.add_row(text, predicted_label, true_label, correct_mark)

    console.print(table)

    if total == 0:
        console.print("\n[yellow]No labeled examples to evaluate.[/yellow]")
        return 0.0

    accuracy = correct / total
    console.print(f"\nRule based accuracy on SAMPLE_POSTS: [bold blue]{accuracy:.2f}[/bold blue]")
    return accuracy


def run_batch_demo() -> None:
    """
    Run the MoodAnalyzer on the sample posts and print predictions only.

    This is a quick way to see how your rules behave without comparing
    to the true labels.
    """
    analyzer = MoodAnalyzer()
    
    table = Table(title="Batch Demo (Rule Based)")
    table.add_column("Text", style="cyan")
    table.add_column("Predicted Label", style="magenta")

    for text in SAMPLE_POSTS:
        label = analyzer.predict_label(text)
        table.add_row(text, label)
    
    console.print(table)


def run_interactive_loop() -> None:
    """
    Let the user type their own sentences and see the predicted mood.

    Type 'quit' or press Enter on an empty line to exit.
    """
    analyzer = MoodAnalyzer()
    console.print("\n=== Interactive Mood Machine (rule based) ===", style="bold cyan")
    console.print("Type a sentence to analyze its mood.")
    console.print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = console.input("[bold green]You:[/bold green] ").strip()
        if user_input == "" or user_input.lower() == "quit":
            console.print("Goodbye from the Mood Machine.", style="bold yellow")
            break

        label = analyzer.predict_label(user_input)
        console.print(f"Model: [bold magenta]{label}[/bold magenta]")


if __name__ == "__main__":
    evaluate_rule_based(SAMPLE_POSTS, TRUE_LABELS)

    run_batch_demo()

    run_interactive_loop()

    print("\nTip: After you explore the rule based model here,")
    print("run `python ml_experiments.py` to try a simple ML based model")
    print("trained on the same SAMPLE_POSTS and TRUE_LABELS.")
