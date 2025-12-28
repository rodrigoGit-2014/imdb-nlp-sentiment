# src/nlp_imdb/data/dataset_contract.md

# Dataset Contract — IMDb Movie Reviews

## Fuente
- Dataset: IMDb Movie Reviews
- Proveedor: Hugging Face (`imdb`)

## Tarea
Clasificación de sentimiento binaria (positivo / negativo).

## Esquema del Dataset Interno

Cada registro del dataset **debe cumplir** con el siguiente formato:

| Campo  | Tipo | Descripción |
|------|------|-------------|
| id | str | Identificador único dentro del proyecto |
| text | str | Texto de la reseña |
| label | int | 0 = negativo, 1 = positivo |
| split | str | train / validation / test |
| source | str | Origen del dataset (hf_imdb) |

### Ejemplo

```json
{
  "id": "train_000001",
  "text": "This movie was absolutely fantastic...",
  "label": 1,
  "split": "train",
  "source": "hf_imdb"
}
