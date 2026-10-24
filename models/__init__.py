from models.base_model import BaseModel
from models.registry import ModelRegistry

# Import all models to trigger decorator registration
import models.cnn
import models.cnn_transformer
import models.cnn_bilstm
import models.cnn_transformer_bilstm
import models.attention_hybrid
import models.resnet1d
import models.inception_time
import models.xresnet1d
import models.convnext1d
import models.moderntcn
import models.inception_time_hybrid

__all__ = ["BaseModel", "ModelRegistry"]
