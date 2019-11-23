```bash
java -Xmx512m -jar reverb-latest.jar headlines_uniq.txt > real_reverb_result.txt
cat real_reverb_result.txt | cut -f2 | uniq > reverb_indicies.txt
```
  
