1. Generate result of reverbs
Note that REVERB sentence order in output is not similar to the input, we use a trick: 


```bash
cat -n headlines_uniq.txt > headlines_uniq_nl.txt
java -Xmx512m -jar reverb-latest.jar headlines_uniq_nl.txt > real_reverb_result.txt
cat real_reverb_result.txt | cut -f1 -d " " | uniq > reverb_uniq.tx
```
  
2. Generate result of sentences with verbs
3. Generate result of sentences with pipeline

```bash
python3 extraction.py
```
