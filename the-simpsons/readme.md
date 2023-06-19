# The Simpsons chatbot

This folder contains the code of a chatbot that uses a dataset of scripts lines from [The Simpsons sitcom](https://en.wikipedia.org/wiki/The_Simpsons)  to answer questions using quotes (from the show).

This application provides a simple example of how to add "context" to a chabot, by prioritizing answers from the same conversation or from the same episode.  
Note that the data wasn't sanitized to remove lines that doesn't make sense without knowing the episode's circumstances, limiting sometimes the accuracy of the chatbot's responses.


## Dialogue example

**You:** Hello!

**Marge Simpson:** Squiggy?  
_(Source = Episode: 296 | Previous line: Hello!)_


**You:** How do you feel today?

**Ned Flanders:** Uh, actually, I'm a little chilly. Could I have another straitjacket?  
_(Source = Episode: 161 | Previous line: Well, how are you feeling this evening?)_


**You:** What did you do today?

**Tammy:** We're gonna have Margaritas and poke gentle fun at our husbands. Would you like to join us?  
_(Source = Episode: 363 | Previous line: What are you doing today?)_


**You:** All normal people love meat

**Bart Simpson:** YOU DON'T WIN FRIENDS WITH SAL-AD / YOU DON'T WIN FRIENDS WITH SAL-AD...  
_(Source = Episode: 133 | Previous line: But all normal people love meat. If I went to a barbecue and there was no meat, I would say, "Yo, Goober! Where's the meat?" I'm trying to impress people here, Lisa. You don't win friends with salad.)_


**You:** I'm Benedict Arnold

**Legs:** The same Benedict Arnold who plotted to surrender West Point to the hated British?  
_(Source = Episode: 118 | Previous line: Benedict Arnold!)_


**You:** Actually my name is Barney Gumbel

**Legs:** The same Barney Gumbel who keeps taking pictures of my sister?  
_(Source = Episode: 118 | Previous line: Uh... actually my name is Barney. Yeah. Barney Gumbel.)_


**You:** Goodbye

**Homer Simpson:** You see, Marge? You see?  
_(Source = Episode: 105 | Previous line: Goodbye.)_


## License

All the code in this folder is free software; you can redistribute it and/or modify it under the terms of the Mozilla Public License v2.0.

The dataset was obtained from https://www.kaggle.com/code/ambarish/fun-in-text-mining-with-simpsons/ is licensed under a Apache 2.0 License.
