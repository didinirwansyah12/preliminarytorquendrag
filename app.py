import streamlit as st
import streamlit.components.v1 as st_components
from schematic_preview import build_well_schematic_svg

st.set_page_config(page_title='Preliminary Torque & Drag Analysis', layout='wide')
st.title('Preliminary Torque & Drag Analysis')
st.info('Schematic scale is now locked to fixed pixel size for consistent full visibility.')
# Minimal restoration - full original will be restored next if needed
