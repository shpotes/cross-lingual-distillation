from datasets import (
  load_dataset,
  list_datasets,
  concatenate_datasets
)

standard_datasets = [
  'bible_para', 
  'kde4', 
  'multi_para_crawl',
  'open_subtitles', 
  'opus_gnome', 
  'opus_paracrawl',
  'opus_ubuntu', 
  'qed_amara',
  'tatoeba',
]

def standard_translation(dataset_name):
  dataset = load_dataset(dataset_name, lang1='es', lang2='eu')['train']

  for col in dataset.column_names:
    if col == 'translation':
      continue

    dataset = dataset.remove_columns(col)

  assert dataset.column_names == ['translation']

  return dataset

def eitb_parcc():
  return load_dataset('eitb_parcc')['train']

def ms_terms():
  pass # TODO: Create huggingface issue

def opus_elhuyar():
  return load_dataset('opus_elhuyar')['train']

def ted_talks_iwslt():
  ted_years = []
  for year in ['2014', '2015', '2016']:
    ted_years.append(load_dataset(
      'ted_talks_iwslt',
      language_pair=('es', 'eu'),
      year=year
    )['train'])

  return concatenate_datasets(ted_years)

def get_dataset():
  big_dataset = []

  for ds_name in standard_datasets:
    big_dataset.append(
      standard_translation(ds_name)
    )

    big_dataset.extend([
      eitb_parcc(), opus_elhuyar(), ted_talks_iwslt()
    ])

  return concatenate_datasets(big_dataset)
