from typing import Dict, Type, Any, List

class ModelRegistry:
    """A global registry to dynamically register and instantiate ECG neural network architectures.

    Used to avoid hardcoded mappings and decouple configuration names from class code.
    """
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a subclass model under a unique string key.

        Args:
            name: The configuration identifier name (e.g. 'cnn_transformer').
        """
        def decorator(subclass: Type):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Instantiates a model registered under the given name.

        Args:
            name: Register name key.
            kwargs: Initialization arguments for the model.

        Returns:
            An instantiated neural network model.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Model architecture '{name}' is not registered. "
                f"Available registered models: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """Lists all registered model names."""
        return list(cls._registry.keys())
