all:
	cp hw3/experiments/$(runid)/predictions-test-$(epoch).npy predictions.npy
	cp hw3/experiments/$(runid)/generated-test-$(epoch).txt generated.txt
	cp hw3/experiments/$(runid)/generated_logits-test-$(epoch).npy generated_logits.npy
	cp hw3/training.ipynb training.ipynb
	tar -cvf handin.tar training.ipynb predictions.npy generated.txt generated_logits.npy
	rm -f generated.txt predictions.npy training.ipynb generated_logits.npy
