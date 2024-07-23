from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import torch
from pyannote.audio import Pipeline
import os
from faster_whisper import WhisperModel
from pydub import AudioSegment
from random import randint
import gradio as gr


model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_sum = T5ForConditionalGeneration.from_pretrained(model_name)


def summ_mT5_G(text):
    input_ids = tokenizer(
        [text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    output_ids = model_sum.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]

    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary


# punctuation
model_punc, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_te')


def punct(text):
    # print(text)
    return apply_te(text.lower(), lan='ru')


pipeline_a = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=str(os.getenv("s1")))


# верификация
def speackers_list(audio_f : str):
    # # send pipeline to GPU (when available)
    # import torch
    # pipeline_a.to(torch.device("cuda"))

    # apply pretrained pipeline
    diarization = pipeline_a(audio_f)
    speackers_list = []
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        speackers_list.append([turn.start, turn.end, speaker])

    # speackers_list.sort(key = lambda x: x[0])
    # print(speackers_list)
    i = 0
    while i<len(speackers_list)-1:
        if speackers_list[i][-1] == speackers_list[i+1][-1] and (i==0 or (speackers_list[i-1][1]<speackers_list[i][1])):
              speackers_list[i][1] = speackers_list[i+1][1]
              speackers_list.pop(i+1)
        else:
          i+=1
    return speackers_list


# speackers = speackers_list(name_of_file)

model_size = "large-v3"
# Run on GPU with FP16
model_tts = WhisperModel(model_size,    device="cpu", compute_type="int8") #,device="cuda", compute_type="float16") #


def speach_to_text(file_name):

    segments, info = model_tts.transcribe(file_name, beam_size=5)

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text_of_seg = ""
    for segment in segments:
        # segment_text = Segment_text(text = segment.text, start = round(segment.start,5), end = round(segment.end, 5))
        # list_of_segments.append(segment_text)
        # print(f"[{segment.start:.2f}  -> {segment.end:.2f}] {segment.text}")

        text_of_seg += segment.text
    # return list_of_segments
    return text_of_seg


class Segment_text:
    text : str # текст сегмента, примененный
    start : float # время начала сегмента
    end : float  # время окончания сегмента
    speacker : str # верифицированный спикер
    summarization : str # абстракт исходного текста сегмента


    def __init__(self, text : str, start : float, end : float, speacker : str ): # , summarization : str):
        self.text = punct(text)
        self.start = start
        self.end = end
        self.speacker = speacker
        self.value = len(text)
        self.summarization = summ_mT5_G(text) if self.value > 200 else text # summarization

    def get_text(self):
        return self.text

    def get_time(self):
        return (self.start, self.end)

    def get_summarization(self):
        return self.summarization

    def get_speacker(self):
        return self.speacker


def init_segments(speackers, name_of_file):
    list_of_segments = []
    audio = AudioSegment.from_file(name_of_file)
    for ind, seg in enumerate(speackers):
        temp_seg = audio[seg[0]*10**3 : seg[1]*10**3]
        name_of_seg = "seg"+str(ind)+".mp3"
        temp_seg.export(name_of_seg, format="mp3")

        temp_text = speach_to_text(name_of_seg)
        segment_text = Segment_text(text = temp_text, start = seg[0], end = seg[1], speacker = seg[2])
        list_of_segments.append(segment_text)
    return list_of_segments


def get_text_to_out(list_of_segments : list):
    res_text, res_sum = "", ""
    for seg in list_of_segments:
        res_text += f"{seg.get_speacker()} : {seg.get_text()}\n"
    for seg in list_of_segments:
        res_sum += seg.get_speacker() + ":  " + seg.get_summarization() + "\n"
    return res_text, res_sum


def do_smth(file):
    audio = AudioSegment.from_wav(file)
    name_of_file = "f"+str(randint(1,10**8))
    audio.export(name_of_file, format="mp3")

    speackers = speackers_list(name_of_file)

    list_of_segments = init_segments(speackers, name_of_file)

    out_text, out_sum = get_text_to_out(list_of_segments)

    return out_text, out_sum


demo = gr.Interface(
    do_smth,
    gr.Audio(type="filepath"),
    [
        gr.Textbox(value="", label="Исходный текст"),
        gr.Textbox(value="", label="Сокращенный текст")
    ]
)
demo.launch()