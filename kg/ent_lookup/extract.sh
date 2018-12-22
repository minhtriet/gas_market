#!/usr/bin/env bash
java -Xmx512m -jar reverb-latest.jar input.txt > output_reverb.txt
cd ../../lib/
java -Xms6g -cp "*" edu.stanford.nlp.naturalli.OpenIE ../kg/ent_lookup/input.txt -output ../kg/ent_lookup/output_stanford_openIE.txt -format ollie
cd -
