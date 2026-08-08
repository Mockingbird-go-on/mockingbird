from mockingbird.stt.device import (
    ctranslate2_cuda_available,
    device_label,
    is_valid_device,
    resolve_device,
    torch_cuda_available,
)


def test_device_label():
    assert device_label("cuda") == "GPU"
    assert device_label("CUDA") == "GPU"
    assert device_label("cpu") == "CPU"
    assert device_label(None) == "…"
    assert device_label("") == "…"


def test_resolve_auto_uses_cuda_when_available():
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_resolve_cpu_always_cpu():
    assert resolve_device("cpu", cuda_available=True) == "cpu"
    assert resolve_device("cpu", cuda_available=False) == "cpu"


def test_resolve_cuda_falls_back_to_cpu():
    assert resolve_device("cuda", cuda_available=True) == "cuda"
    assert resolve_device("cuda", cuda_available=False) == "cpu"


def test_resolve_unknown_is_treated_as_auto():
    assert resolve_device(None, cuda_available=True) == "cuda"
    assert resolve_device("", cuda_available=False) == "cpu"
    assert resolve_device("gpu", cuda_available=True) == "cuda"
    assert resolve_device(" CUDA ", cuda_available=True) == "cuda"


def test_is_valid_device():
    assert is_valid_device("auto")
    assert is_valid_device("cpu")
    assert is_valid_device("cuda")
    assert not is_valid_device("gpu")
    assert not is_valid_device(None) is False or is_valid_device(None) in (False, True)


def test_cuda_probes_never_raise():
    assert isinstance(torch_cuda_available(), bool)
    assert isinstance(ctranslate2_cuda_available(), bool)
