"""Factories for separately verified official external baselines, never training."""


def load_detector(model, model_dir):
    if model == 'picodet':
        from external_picodet import PicoDetDetector
        return PicoDetDetector(model_dir)
    if model in ('blazeface', 'blazeface_s5_size'):
        from external_blazeface import BlazeFaceDetector
        return BlazeFaceDetector(model_dir, input_size=(384, 256) if model.endswith('_s5_size') else None)
    raise ValueError('Unknown frozen baseline: ' + model)
