from typing import Literal
from pydantic import BaseModel
import torch
from deepvqe import DeepVQE
from infer import STFT, load_wav_stereo, _match_lengths, make_fade_window, run_one_chunk, _pad_or_trim_to_chunk, \
    save_wav_stereo
from pathlib import Path



class Model:
    def __init__(
            self,
            ckpt: str|Path,
            device: str = 'cuda',
    ):
        self.ckpt = ckpt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ckpt args
        self.sr = None
        self.n_fft = None
        self.hop = None
        self.win = None
        self.delay_past_frames = None
        self.delay_future_frames = None
        self.align_hidden = None
        self.epoch = None

        # models
        self.model = None
        self.stft_mod = None

    def load_model(self):
        ckpt = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {}) or {}
        self.sr = int(ckpt_args.get("sr", 48000))
        self.n_fft = int(ckpt_args.get("n_fft", 1536))
        self.hop = int(ckpt_args.get("hop", 480))
        self.win = int(ckpt_args.get("win", 1536))
        self.delay_past_frames = int(ckpt_args.get("delay_past_frames", 25))
        self.delay_future_frames = int(ckpt_args.get("delay_future_frames", 25))
        self.align_hidden = int(ckpt_args.get("align_hidden", 64))
        self.epoch = int(ckpt.get("epoch", 0) or 0)

        self.model = DeepVQE(
            n_fft=self.n_fft,
            delay_past_frames=self.delay_past_frames,
            delay_future_frames=self.delay_future_frames,
            align_hidden=self.align_hidden
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        self.stft_mod = STFT(
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win
        ).to(self.device)

    def process_audio(
            self,
            mix_path: str|Path,
            ref_path: str|Path,
            out_path: str|Path,
            *,

            amp: bool = False,
            amp_dtype: Literal["bf16", "fp16", "tf32"] = 'fp16',
            length_mode: Literal["crop", "pad"] = 'crop',
            chunk_sec: int = 8,
            overlap_sec: int = 2,
    ):
        mix = load_wav_stereo(mix_path, self.sr).to(self.device)
        ref = load_wav_stereo(ref_path, self.sr).to(self.device)

        if mix.shape[1] != ref.shape[1]:
            print(f"[warn] length mismatch: mix={mix.shape[1]} ref={ref.shape[1]} | mode={length_mode}")
        mix, ref = _match_lengths(mix, ref, mode=length_mode)

        C, T = mix.shape
        assert C == 2

        chunk_len = int(round(chunk_sec * self.sr))
        overlap_len = int(round(overlap_sec * self.sr))
        if overlap_len >= chunk_len:
            raise RuntimeError("overlap-sec must be smaller than chunk-sec")
        if chunk_len <= 0:
            raise RuntimeError("chunk-sec too small")
        if overlap_len < 0:
            raise RuntimeError("overlap-sec must be >= 0")

        step = chunk_len - overlap_len
        fade = make_fade_window(chunk_len, overlap_len, device=self.device)

        out = torch.zeros((2, T), device=self.device, dtype=torch.float32)
        wsum = torch.zeros((T,), device=self.device, dtype=torch.float32)

        use_amp = amp and (self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16 # FIXME

        pos = 0
        while pos < T:
            end = pos + chunk_len
            mix_chunk = mix[:, pos:end]
            ref_chunk = ref[:, pos:end]

            # ensure EXACT chunk_len for both
            mix_chunk = _pad_or_trim_to_chunk(mix_chunk, chunk_len)
            ref_chunk = _pad_or_trim_to_chunk(ref_chunk, chunk_len)

            out_chunk = run_one_chunk(self.model, self.stft_mod, mix_chunk, ref_chunk, use_amp, amp_dtype)  # (2,chunk_len)

            valid_len = min(chunk_len, T - pos)
            out[:, pos: pos + valid_len] += out_chunk[:, :valid_len] * fade[:valid_len]
            wsum[pos: pos + valid_len] += fade[:valid_len]

            pos += step

        out = out / (wsum.clamp_min(1e-8)[None, :])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_wav_stereo(out_path, out, self.sr, subtype='FLOAT')
        print("saved:", out_path)
        mix.to('cpu')
        ref.to('cpu')
        fade.to('cpu')
        out.to('cpu')
        wsum.to('cpu')
        del mix
        del ref
        del fade
        del out
        del wsum



ckpt_base_path = Path('ckpt_48k').resolve()


ckpts = [ckpt_base_path / f'deepvqe_aec48k_e{str(e).zfill(3)}.pt' for e in range(3, 5)]

models = [Model(ckpt) for ckpt in ckpts]



class TestCase(BaseModel):
    base_name: str
    mix_path: Path
    ref_path: Path
    full_name: str


def get_tests_samples(tests_dir: Path):
    test_cases: list[TestCase] = []

    for sample in tests_dir.iterdir():
        base_name = sample.name
        mix_path = sample / 'mix.wav'
        ref_path = sample / 'ref.wav'
        if not mix_path.exists():
            raise FileNotFoundError(mix_path)
        if not ref_path.exists():
            raise FileNotFoundError(ref_path)
        full_name = f'{base_name}.wav'

        case_obj = TestCase(
            base_name=base_name,
            mix_path=mix_path,
            ref_path=ref_path,
            full_name=full_name,
        )
        test_cases.append(case_obj)
    return test_cases


samples = get_tests_samples(Path('tests').resolve())


for i in range(len(models)):
    model: Model= models.pop(0)
    model.load_model()

    for sample in samples:
        mix_path = sample.mix_path
        ref_path = sample.ref_path

        out_path = Path('tests_out').resolve() / str(model.epoch) / sample.full_name

        model.process_audio(
            mix_path=mix_path,
            ref_path=ref_path,
            out_path=out_path,
        )
    del model





