from src.models.providers import cloud_api

def get_model(config: dict = None, provider: str = None, model_name: str = None, temperature: float = 0.7):
    # Если передан config, используем его
    if config:
        provider = config["provider"]
        model_name = config["model_name"]
        temperature = config.get("temperature", 0.7)
    
    # Проверяем что параметры заданы
    if not provider or not model_name:
        raise ValueError("Необходимо указать provider и model_name")
    
    if provider == "cloud":
        return cloud_api.CloudAPIModel(
            model_name=model_name,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
