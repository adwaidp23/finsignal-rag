# This directory stores the committed FAISS index files.
# 
# Files in this directory:
#   index.faiss  — FAISS binary index (committed to git)
#   index.pkl    — LangChain FAISS docstore pickle (committed to git)
#
# How to update the index:
#   Run: python scripts/build_index.py
#   Then: git add data/faiss_index/ && git commit -m "rebuild index"
#
# NOTE: If the index grows > 50 MB, switch to Git LFS:
#   git lfs track "data/faiss_index/*.faiss"
