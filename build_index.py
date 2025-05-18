from graph import LangGraphClass
from config import settings


graph = LangGraphClass(settings)

graph.create_vector_index(settings.INDEX_PATH, settings.CSV_DATA_PATH)
