"""Streamlit Community Cloud entry point. Local demo is unaffected."""
import os
from pathlib import Path
import subprocess
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.environ['ENCODER_CLOUD'] = '1'
os.environ['ENCODER_SKIP_CLIP'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, str(ROOT / 'demo'))


@st.cache_resource
def prepare():
    subprocess.run([sys.executable, str(ROOT / 'scripts/setup_demo.py')],
                   cwd=ROOT, check=True)
    import torch
    torch.set_num_threads(2)
    return True


try:
    with st.spinner('Preparing model assets. First startup may take several minutes...'):
        prepare()
except (OSError, subprocess.CalledProcessError):
    st.error('Model setup failed. Check deployment logs and reboot the app to retry.')
    st.stop()

navigation = st.navigation([
    st.Page(str(ROOT / 'demo/🔍_Demo_Retrieval.py'), title='Retrieval', default=True),
    st.Page(str(ROOT / 'demo/pages/1_📖_About.py'), title='About'),
    st.Page(str(ROOT / 'demo/pages/2_💬_Ask_AI.py'), title='Ask AI'),
])
navigation.run()
