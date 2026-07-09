import numpy as np
import soundfile as sf

import main


def test_load_assets_resolves_model_files_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    main.MODEL = None
    main.SCALER = None
    main.IQR_BOUNDS = None
    main.EXPECTED_FEATURES = 16

    main.load_assets()

    assert main.MODEL is not None
    assert main.SCALER is not None
    assert main.IQR_BOUNDS is not None


def test_predict_from_audio_bytes_returns_prediction(tmp_path):
    audio_path = tmp_path / "sample.wav"
    sr = 16000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(audio_path, y, sr)

    main.load_assets()
    result = main.predict_from_audio_bytes(audio_path.read_bytes())

    assert "prediction" in result
    assert result["prediction"] in {0, 1}
    assert result["total_chunks_analyzed"] >= 1
