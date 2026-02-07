"""
Workflow orchestration using Composite and Builder patterns.

Workflow: Composes multiple PipelineStep objects and executes them
sequentially, passing a shared context dictionary.

WorkflowBuilder: Fluent API for constructing workflows step-by-step.
"""

from typing import Any, Dict, List, Optional, Self

from src.pipeline.steps import PipelineStep


class Workflow:
    """Orchestrates execution of multiple pipeline steps.

    Composite Pattern: Treats a collection of PipelineSteps as a single
    unit that can be executed together. Each step receives and updates
    a shared context dictionary.

    Example:
        >>> workflow = Workflow(steps=[load_step, train_step, eval_step])
        >>> result = workflow.run()
        >>> print(result['metrics'].accuracy)
    """

    def __init__(self, steps: List[PipelineStep], name: str = "Workflow") -> None:
        """Initialize with a list of pipeline steps.

        Args:
            steps: Ordered list of PipelineStep objects.
            name: Human-readable name for the workflow.
        """
        self.steps = steps
        self.name = name

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all steps in sequence.

        Args:
            initial_context: Optional starting context dictionary.
                            Merged with empty dict if provided.

        Returns:
            Final context dictionary after all steps complete.
        """
        context = initial_context.copy() if initial_context else {}

        print(f"\n{'='*60}")
        print(f"Starting Workflow: {self.name}")
        print(f"{'='*60}")

        for i, step in enumerate(self.steps, 1):
            print(f"\n[Step {i}/{len(self.steps)}] {step.get_name()}")
            print("-" * 40)
            context = step.execute(context)

        print(f"\n{'='*60}")
        print(f"Workflow '{self.name}' completed successfully!")
        print(f"{'='*60}\n")

        return context

    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the workflow.

        Args:
            step: PipelineStep to append.
        """
        self.steps.append(step)

    def __len__(self) -> int:
        """Return number of steps."""
        return len(self.steps)


class WorkflowBuilder:
    """Fluent builder for constructing Workflow objects.

    Builder Pattern: Enables step-by-step construction of workflows
    with a fluent API that chains method calls.

    Example:
        >>> workflow = (WorkflowBuilder()
        ...     .set_name("Training Pipeline")
        ...     .add_step(LoadDataStep(config))
        ...     .add_step(TrainModelStep(config))
        ...     .add_step(EvaluateModelStep())
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize an empty builder."""
        self._steps: List[PipelineStep] = []
        self._name: str = "Workflow"

    def set_name(self, name: str) -> Self:
        """Set the workflow name.

        Args:
            name: Human-readable workflow name.

        Returns:
            self for method chaining.
        """
        self._name = name
        return self

    def add_step(self, step: PipelineStep) -> Self:
        """Add a step to the workflow being built.

        Args:
            step: PipelineStep to add.

        Returns:
            self for method chaining.
        """
        self._steps.append(step)
        return self

    def build(self) -> Workflow:
        """Build and return the configured Workflow.

        Returns:
            Configured Workflow instance.

        Raises:
            ValueError: If no steps have been added.
        """
        if not self._steps:
            raise ValueError("Cannot build workflow with no steps. Add steps first.")

        return Workflow(steps=self._steps.copy(), name=self._name)

    def reset(self) -> Self:
        """Reset the builder to start fresh.

        Returns:
            self for method chaining.
        """
        self._steps = []
        self._name = "Workflow"
        return self
