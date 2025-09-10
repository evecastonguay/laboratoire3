"""
Comment régler le problème concernant l'importation des packages? (import error)

1) le problème est dans l'environnement virtuel (venv), configuré pour puce intel et non apple silicon (x86_64 vs arm64).
2) aller dans terminal, supprimer venv du projet:
cd /Users/evecastonguay/PycharmProjects/Labo3
rm -rf venv
3) réinstaller un environnement:
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
4) vérifier que bonne puce:
python -c "import platform; print(platform.machine())" (devrait être arm64)
5) réinstaller les bons packages
pip install --no-cache-dir numpy matplotlib netCDF4 [xarray, etc.]
"""