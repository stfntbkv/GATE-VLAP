import tiktoken
def count_tokens(text, model="gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


string = '''You are the humanoid service robot robotXYZ assisting a human in a kitchen. Your task is to turn the user request into a goal state in PDDL. The predicates of the domain are defined as follows:
(:predicates
; Describes ?o being on top of the supporting ?s
(on  ?o - carry ?s - support)
; Do not use this predicate
(indirect_on  ?o - carry ?s - support)
; Describes the non liquid ?o being in the container ?c
(in  ?o - carry ?c - contain)
; Describes the liquid ?li being in the container ?lc
(liquid_in  ?li - liquid ?lc - liquid-contain)
; Describes the actor ?a being at the location ?l which can also be another actor
(at  ?a - actor ?l - location)
; Describes the object ?o being in the hand of a human or robot ?a. A human can have multiple objects in their hand
(inhand  ?o - carry ?a - actor)
; Describes the object ?ac being carried together by the actors ?a and ?r
(carried  ?ac - assisted-carry ?a - actor ?r - actor)
; Do not use this predicate
(free  ?s - support)
; Do not use this predicate
(reachable  ?o - carry ?l - location ?a - actor)
; Describes that ?op is open
(opened  ?op - open)
; Describes that ?cl is closed
(closed  ?cl - close)
(warm  ?co - consumable)
(liquid_warm  ?li - liquid)
(wet  ?ws - wet-swipe)
; Describes that a consumable ?co is chopped
(chopped  ?co - consumable)
; Describes that an object ?ob has been cleaned with water
(clean  ?ob - object)
; Describes that the robot has visited the given location ?l
(visited  ?l - location)
; Do not use this predicate
(hand_occupied  ?a - actor)
) 
 and the problem without the goal is:
 (define (problem test)

    (:domain robotic_planning)
    
    (:objects
        sink0 - sink
        cucumber0 - cucumber
        window0 - window
        human0 - human
        box0 - box
        spatula0 - spatula
        plastic_cup1 - plastic_cup
        watering_can0 - watering_can
        cutting_board0 - cutting_board
        spoon0 - spoon
        kitchen_knife0 - kitchen_knife
        cornflakes0 - cornflakes
        sponge0 - sponge
        milk0 - milk
        apple0 - apple
        tissue0 - tissue
        milk_cartoon0 - milk_cartoon
        water0 - water
        banana0 - banana
        table0 - table
        salad0 - salad
        bowl0 - bowl
        plate0 - plate
        counter0 - counter
        hammer0 - hammer
        plastic_cup0 - plastic_cup
        robot0 - robot_profile
        bottle0 - bottle
        refrigerator0 - refrigerator
        tomato0 - tomato
        plastic_cup2 - plastic_cup
        door0 - door
    )
    
    (:init 
        (on  watering_can0 counter0)
        (on  spoon0 counter0)
        (on  cutting_board0 table0)
        (in  cucumber0 refrigerator0)
        (on  kitchen_knife0 counter0)
        (on  cornflakes0 counter0)
        (in  hammer0 box0)
        (closed  refrigerator0)
        (closed  bottle0)
        (on  apple0 table0)
        (on  sponge0 counter0)
        (at  human0 table0)
        (liquid_in  milk0 milk_cartoon0)
        (closed  door0)
        (on  tissue0 counter0)
        (on  milk_cartoon0 table0)
        (closed  window0)
        (on  banana0 table0)
        (on  plate0 table0)
        (on  bowl0 table0)
        (on  plastic_cup0 table0)
        (on  bottle0 table0)
        (opened  milk_cartoon0)
        (liquid_in  water0 bottle0)
        (on  plastic_cup2 table0)
        (in  salad0 refrigerator0)
        (in  tomato0 bowl0)
        (on  plastic_cup1 table0)
        (on  spatula0 counter0)
        (at  robot0 table0)
        (on  box0 counter0)
        (= total-cost 0)
        (= (cost human0) 100)
        (= (cost robot0) 1)
    )
    
    (:metric minimize (total-cost))
    
)
 
 The user asks "Help me prepare a salad with tomatoes". Explain step by step how you concluded that this is the desired goal state. Express the goal state in correct PDDL Syntaxwithout using the ? operator. You are allowed to use disjunctions
These are examples from previous user interactions:
Q1: Please assist me in making a salad with tomatoes
A: (:goal and (chopped tomato0) (chopped salad0) (in salad0 bowl0) (in tomato0 bowl0))
Consider these example in your answer'''
string2 = '''To determine the desired goal state for the user request "Help me prepare a salad with tomatoes," we need to analyze the request and map it to the predicates and objects defined in the PDDL domain. Here's the step-by-step process:

1. **Understanding the Request**: The user wants to prepare a salad that includes tomatoes. This implies that both the salad and the tomatoes need to be in a state suitable for consumption as part of a salad.

2. **Identifying Relevant Predicates**:
   - `(chopped ?co - consumable)`: This predicate indicates that a consumable item (like a tomato or salad) has been chopped, which is a typical preparation step for these ingredients in a salad.
   - `(in ?o - carry ?c - contain)`: This predicate indicates that an object (like chopped tomatoes or salad) is inside a container (like a bowl), which is necessary for mixing or serving the salad.

3. **Identifying Relevant Objects**:
   - `tomato0`: Represents the tomatoes.
   - `salad0`: Represents the salad greens.
   - `bowl0`: Represents the bowl in which the salad will be mixed or served.

4. **Mapping the Request to Predicates and Objects**:
   - The tomatoes (`tomato0`) need to be chopped and placed in the bowl (`bowl0`).
   - The salad (`salad0`) also needs to be chopped and placed in the same bowl (`bowl0`).

5. **Formulating the Goal State**:
   - We need to ensure that both `tomato0` and `salad0` are chopped.
   - Both `tomato0` and `salad0` need to be inside `bowl0`.

Given this analysis, the goal state in PDDL syntax can be expressed as follows:

```PDDL
(:goal
    (and
        (chopped tomato0)
        (chopped salad0)
        (in tomato0 bowl0)
        (in salad0 bowl0)
    )
)
```

This goal state ensures that both the tomatoes and the salad are prepared (chopped) and placed in the bowl, ready for serving or further dressing, aligning with the user's request to help prepare a salad with tomatoes.
'''
print(count_tokens(string)+count_tokens(string2))