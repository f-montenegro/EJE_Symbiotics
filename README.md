# Credit Rating: Credit Metrics model application

This repository contains the python code referent to the tasks of Mission 2622 provided by EJE (ENSAE Junior Etudes) to the company Symbiotics.

This mission consisted of the development of a Credit Rating Stress Test method inspired by an open source model known as Credit Metrics, initially developed by the research team of JP Morgan.

I invite you to the <a href="https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:2e1301a0-ea68-3e2b-bd25-1350c12ee3f7">Report of the Mission</a> for further details about the mission and development of this python tool.

 - **Layer 0 (C0):** The scripts of layer 0 are used to call all the python libraries used in the code. Thus, all layer 1 code starts with an "import" of layer 0 files. In addition, layer 0 contains the requirement.txt file, a file used to manage package dependencies and its respective versions to make possible the execution of the set of scripts;
 
- **Layer 1 (C1):** After importing all the libraries stated in the Layer 0, the environment is ready for development. Therefore, Layer 1 contains all the classes/functions used on the project. There is a total of three classes, the first two prepare the inputs (i.e. they generate the Correlation Matrix and the Transition Matrix). The last one is the Credit Metrics model class containing the functions used to generate the final output (Credit Metrics);

- **Layer 2 (C2):** Finally, Layer 2 contains all the interesting files for the main user1. The first one is the script "Variables.py". As the name says, it is the place where the user will set all the desired parameters for the model2. The second one is the script "main.py", which is the file that, when executed, generates the final output3. For last, the "main.py" code creates a folder containing every output: the set of Transition Matrix and Correlation Matrix, and, obviously, the Credit Metrics output.

**_Attention: The user just need to change the variables in the file Variables.py and execute
the file main.py to obtain the final output._**