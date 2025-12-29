# src/nlp_imdb/data/dataset_contract.md

# Dataset Contract — IMDb Movie Reviews

## Fuente
- Dataset: IMDb Movie Reviews
- Proveedor: Hugging Face (`imdb`)
- Loader del proyecto: `nlp_imdb.data.dataset_loader.load_imdb_hf`

## Tarea
Clasificación de sentimiento binaria (positivo / negativo).

## Esquema del Dataset Interno (Contract)

Cada registro del dataset **debe cumplir** con el siguiente formato lógico:

| Campo  | Tipo | Descripción |
|------|------|-------------|
| id | str | Identificador único dentro del proyecto (si se materializa a disco) |
| text | str | Texto de la reseña |
| label | int | 0 = negativo, 1 = positivo |
| split | str | train / validation / test |
| source | str | Origen del dataset (hf_imdb) |

> Nota: En la carga directa desde Hugging Face (DatasetDict), el contract se garantiza por:
> - splits presentes: `train`, `validation`, `test`
> - columnas presentes: `text`, `label`

### Ejemplo (si se exporta a formato tabular/JSON)

```json
{
  "id": "train_000001",
  "text": "This movie was absolutely fantastic...",
  "label": 1,
  "split": "train",
  "source": "hf_imdb"
}
