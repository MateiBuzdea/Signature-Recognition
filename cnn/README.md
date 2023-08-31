# One-Class classification using CNN
This model is trained to recognize only one of the many training signatures as
valid, while the rest are marked as invalid.

Because the number of samples is imbalanced, the class weights must be adjusted
at the beginning.

In order to prevent blank image backdoor (blank images getting recognized as
valid signatures), I created the random color inverse function and I passed it
to ImageDataGenerator.