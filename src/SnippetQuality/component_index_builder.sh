index() {
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $1 \
  --index $2 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --storePositions --storeDocvectors --storeRaw
}

test_collection=ntcir15
for dir in ../../data/index/$test_collection/component_collection/*/
do
    if [ -d $dir ]; then
        filename=$(basename $dir)
        output=../../data/index/$test_collection/component_index/$filename
        mkdir -p $output
        index $dir $output
    fi
done
