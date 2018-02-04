Singular Value Decomposition Experiments on EMG data
=============
These are some experiments done as exploratory research on my senior project.
This is largely brainstorming and mucking about, so there's likely to be little
value or worth in this, and the ideas contained within should be treated with the
playfulness they deserve.


First Idea
---------------
Look at blob of some amount of raw vectors, using ideas from ICA, try to compare it with other
generalizations of blobs. Use blob words, blob dotproducts, or some other poor interpretation
of an idea. Abuse of language is bound to be present in great abundance. 

Training:
normalize, make all input rows into 
Take the singular value decomposition of all raw classes
preserve these vectors

Predicting:
normalize input array, 
perform singular value decomposition on input array
take dot product of input's singular vectors multiplied by their singular values with the learned sing. 

Given n classes, we 


![picture alt](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Emoji_u1f64b.svg/128px-Emoji_u1f64b.svg.png)

Happy lil blobby
