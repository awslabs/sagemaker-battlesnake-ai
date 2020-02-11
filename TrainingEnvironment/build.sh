echo "Create the entry points"
mkdir battlesnake_src
cp -a battlesnake_gym/examples/. battlesnake_src
cp battlesnake_gym/requirements.txt battlesnake_src/

echo "Create the notebook"
cp battlesnake_gym/notebooks/* .