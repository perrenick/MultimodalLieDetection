from PreTrainedAudioClassifier.parse_cremad import make_dataframe
import pandas as pd

OUTPUT_DIR = './AudioWAV_Denoised'

labels_df = make_dataframe(OUTPUT_DIR)
opensmile_df = pd.read_csv('merged_audio_wav_features.csv')

opensmile_df = opensmile_df.drop(columns="name_field", inplace=True)
opensmile_df = opensmile_df.dropna(axis=1, inplace=True)
final_df = pd.concat(opensmile_df, labels_df)