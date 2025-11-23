"""Quick script to check collection information."""
from src.shared.embedding.qdrant_store import QdrantStore
from configs.settings import settings

store = QdrantStore()

print(f"Collection: {settings.embed_collection}")
print()

try:
    info = store.client.get_collection(settings.embed_collection)
    print(f"✅ Collection found!")
    print(f"   Points: {info.points_count}")
    print(f"   Vectors: {info.vectors_count}")
    print(f"   Vector size: {info.config.params.vectors.size}")
    print(f"   Distance metric: {info.config.params.vectors.distance}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

