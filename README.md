
# Encodec and DAC Wrappers

## 1. Quickstart: Running the Example Notebooks

1. **Create a virtual environment and activate it (recommended):**

		python3 -m venv venv_audio_codec_wrappers
		source venv_audio_codec_wrappers/bin/activate

2. **Install requirements:**
	
        pip install -r ../requirements.txt

3. **Run the example notebooks:**
		- Open any notebook in the `examples/` folder (e.g., `example_dac.ipynb`, `example_encodec.ipynb`) using Jupyter or VS Code.
		- Run the cells to test the audio compression and reconstruction pipelines.

---

## 2. Wrapper Methods: Latents and Codes

The goal of these wrappers is to make the full pipeline of both Encodec and DAC open. A list of the methods can be found below. Note that in different cases, inputs might need extra metadata (see example notebook).


- `audio_to_latents()`
- `latents_to_codes()`
- `codes_to_latents()`
- `latents_to_audio()`


---

