![octohand](https://github.com/SioKCronin/akko/blob/master/media/akko.jpg)

# OctoHand

OctoHand is a cephalapod-inspired OpenAI gym-style testing environment for reinforcement learning policy development and soft robotics research. In brief, a simulated robot hand—with tentacles! 

## Inspiration

OctoHand is a cephalapod-inspired OpenAI gym-style testing environment for reinforcement learning policy development and soft robotics research. The physics of the hand itself is inspired by Festo’s [OcotopusGripper](https://www.festo.com/group/en/cms/12745.htm), which is a silicone encased, pneumatically controlled “bionic gripper”. It connects to a compressed air generator, which causes it to contract and grip objects. 

OctoHand will simulate the basic physics of a single Festo tentacle, as well as the functionality of a simple motion terminal allowing the positioning of the hand in relationship to objects and sequencing of motions. The aim will be to develop a series of three testing environments along the lines of what OpenAI has presented with their HandManinpulate series (block, egg, pen), where the task is to develop a policy that will grip the given object and manipulate it into a specific position. 

In my beta phase I aim to implement a three-tentacle hand, and, depending on feedback I get from the system and colleagues, I will explore the feasibility of adding additional tentacles. My plan is to work within the physical constraints of this industry gripper, which has documented in the work of Dr. Li Wen et al. at the School of Mechanical Engineering and Automation at Beihang University (Festo’s research partner). 

While drawing inspiration from nature in robotics is not new, drawing inspiration from a non-human to advance RL policy research in an open-source robotics simulation collaborative environment is. I think other RL researchers will benefit from testing their algorithms on a novel environment, and, more broadly, I believe OctoHand can stoke fresh visions for solving problems in research and industry, as it asks to consider embodied intelligence beyond our anthropic bias. 

Festo has been seeking future technologies through biomimicry, mainly through their Bionic Learning Network, yet the design specifications and code for this research is not open source. My goal with this project is to contribute to 

Like the other environments in OpenAI’s gym, OctoHand will be written in Python will physics provided by Mujoco. As the project develops, I may chose to switch things over to ROS, which is also open source, and in either case I plan to open up the project to collaborators from both the Open Robotics and OpenAI communities. 

In MIRI’s recent Alignment for Advanced Machine Learning Systems publication, a research topic was articulated that I believe we will see branching in coming years - robust human imitation, which asked, how can we design and train ML systems to effectively imitate humans who are engaged in complex and difficult tasks? When it comes to complex and difficult physical tasks, humans are outranked by other species in several categories (fastest land speed - cheetah, fastest ability to transform body into color and textures of surroundings - octopus sweeps the competition, etc.). The field of robotics is continually presented with new and complex physical problems to solve, and I believe OctoHand can play a role in shining light on what nature-inspired soft robotics, and accompanying RL policies, can bring to the table. 
