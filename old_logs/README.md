1. Generate result of reverbs

```bash
java -Xmx512m -jar reverb-latest.jar headlines_uniq.txt > real_reverb_result.txt
cat real_reverb_result.txt | cut -f2 | uniq > reverb_indicies.txt
```
  
2. Generate result of sentences with verbs
3. Generate result of sentences with pipeline

```bash
python3 extraction.py
```
