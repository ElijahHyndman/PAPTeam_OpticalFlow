#! /bin/bash
# Convert all .PNGs in the source folders inside ./images/ to .JPGs

IN="./MPI_images"
LIST=$(ls "$IN")

for ENTRY in $LIST
do
  if [ -d "$IN/$ENTRY" ]
  then # Directory

      DIR="$IN/$ENTRY"
      PNGS="$(ls $DIR | grep png)"
      # Convert PNGs to JPGs
      for PNG in $PNGS
      do
        echo "deleting $DIR/$PNG"
        rm $DIR/$PNG
      done
  else # File
      echo "[ignoring]: $ENTRY"
  fi
done
