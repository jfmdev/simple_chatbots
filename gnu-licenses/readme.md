# GNU Licenses chatbot

This folder contains the code of a chatbot, implemented with sentence transformers, and trained with the [GNU's Frequently Asked Questions site](https://www.gnu.org/licenses/gpl-faq.html), to answer questions about GNU Licenses.

This application is a proof-of-concept showing how to use information from a website to train a chatbot.  
A limitation of this approach is that the chatbot don't has question variations, which could help him to better recognize user questions, nor answer variations, that could make their responses appear more natural to users.


## Dialogue example

**You:** Can I sell software licensed under GPL?

**Chatbot:** Yes, the GPL allows everyone to do this. The right to sell copies is part of the definition of free software. Except in one special situation, there is no limit on what price you can charge. (The one exception is the required written offer to provide source code that must accompany binary-only release.)  
_(source: https://www.gnu.org/licenses/gpl-faq.html#DoesTheGPLAllowMoney | Does the GPL allow me to sell copies of the program for money?)_

**You:** Is the BSD license compatible with GPL?

**Chatbot:** Because it imposes a specific requirement that is not in the GPL; namely, the requirement on advertisements of the program. Section 6 of GPLv2 states:  
You may not impose any further restrictions on the recipients' exercise of the rights granted herein.
GPLv3 says something similar in section 10. The advertising clause provides just such a further restriction, and thus is GPL-incompatible.  
The revised BSD license does not have the advertising clause, which eliminates the problem.  
_(source: https://www.gnu.org/licenses/gpl-faq.html#OrigBSD | Why is the original BSD license incompatible with the GPL?)_

**You:** Does v3 and v2 from GPL are compatible between them?

**Chatbot:** No. Many requirements have changed from GPLv2 to GPLv3, which means that the precise requirement of GPLv2 is not present in GPLv3, and vice versa. For instance, the Termination conditions of GPLv3 are considerably more permissive than those of GPLv2, and thus different from the Termination conditions of GPLv2.  
Due to these differences, the two licenses are not compatible: if you tried to combine code released under GPLv2 with code under GPLv3, you would violate section 6 of GPLv2.
However, if code is released under GPL “version 2 or later,” that is compatible with GPLv3 because GPLv3 is one of the options it permits.  
_(source: https://www.gnu.org/licenses/gpl-faq.html#v2v3Compatibility | Is GPLv3 compatible with GPLv2?)_


## License

All the code in this folder is free software; you can redistribute it and/or modify it under the terms of the Mozilla Public License v2.0.

The dataset is licensed under a Creative
Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0) license.
