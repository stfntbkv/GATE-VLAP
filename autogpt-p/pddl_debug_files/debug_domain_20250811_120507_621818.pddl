(define (domain robotic_planning)
    
    (:requirements :strips :typing :negative-preconditions :existential-preconditions :equality :derived-predicates :disjunctive-preconditions)
    
    (:types
        id grasp carry assisted-carry cut contain liquid-contain enclosed-contain pour precise-pour drink constrained-move rotate axis-roll free-roll push pull open close support stack sturdy-support vertical-support scoop stir distance-connect contact-connect pierce pick hit pound swing dry-swipe wet-swipe heat heat-resistance liquid drinkable consumable location dummy - object
        actor human counter trash_can table - location
        robot human take_outer putiner choper filler heat_liquider moveer wet_swipeer stacker placeer opener handoverer grasper pourer receive_objecter closeer - actor
        lime_soda apple trash_can red_bull pepsi coke tea energy_bar sponge multigrain_chips 7up grapefruit_soda water sprite jalapeno_chips - grasp
        lime_soda apple trash_can red_bull pepsi coke tea energy_bar sponge multigrain_chips 7up grapefruit_soda water sprite jalapeno_chips - carry
        lime_soda trash_can red_bull pepsi coke tea sponge 7up grapefruit_soda water sprite - contain
        lime_soda red_bull pepsi coke 7up grapefruit_soda water sprite - liquid-contain
        lime_soda 7up water sprite - enclosed-contain
        lime_soda red_bull pepsi coke 7up grapefruit_soda water sprite - pour
        lime_soda red_bull pepsi coke 7up grapefruit_soda water sprite - drink
        lime_soda counter apple trash_can red_bull pepsi coke tea energy_bar sponge multigrain_chips table 7up grapefruit_soda water sprite jalapeno_chips - rotate
        lime_soda apple red_bull pepsi coke tea energy_bar sponge 7up grapefruit_soda water sprite - axis-roll
        lime_soda apple trash_can red_bull pepsi coke tea energy_bar sponge multigrain_chips table 7up grapefruit_soda water sprite jalapeno_chips - push
        lime_soda apple trash_can red_bull pepsi coke tea energy_bar sponge multigrain_chips table 7up grapefruit_soda water sprite jalapeno_chips - pull
        lime_soda trash_can red_bull pepsi coke 7up grapefruit_soda water sprite - open
        lime_soda trash_can pepsi coke 7up grapefruit_soda water sprite - close
        lime_soda red_bull pepsi coke 7up grapefruit_soda water sprite - stir
        counter table - support
        counter trash_can red_bull pepsi coke energy_bar sponge multigrain_chips grapefruit_soda jalapeno_chips - stack
        counter trash_can table - vertical-support
        apple - free-roll
        apple red_bull coke tea energy_bar multigrain_chips grapefruit_soda sprite jalapeno_chips - consumable
        tea - liquid
        tea 7up sprite - drinkable
        sponge - wet-swipe
        table - assisted-carry
        sprite - precise-pour
        robot_profile - grasper
        robot_profile - placeer
        robot_profile - take_outer
        robot_profile - putiner
        robot_profile - handoverer
        robot_profile - receive_objecter
        robot_profile - moveer
        robot_profile - pourer
        robot_profile - filler
        robot_profile - opener
        robot_profile - closeer
        robot_profile - choper
        robot_profile - heat_liquider
        robot_profile - wet_swipeer
        robot_profile - stacker
        robot_profile - robot
    )
    
    (:predicates
        (on  ?o0 - object ?o1 - object)
        (in  ?o0 - object ?o1 - object)
        (liquid_in  ?o0 - object ?o1 - object)
        (at  ?o0 - object ?o1 - object)
        (inhand  ?o0 - object ?o1 - object)
        (carried  ?o0 - object ?o1 - object ?o2 - object)
        (empty  ?o0 - object)
        (indirect_on  ?o0 - object ?o1 - object)
        (reachable  ?o0 - object ?o1 - object ?o2 - object)
        (free  ?o0 - object)
        (opened  ?o0 - object)
        (closed  ?o0 - object)
        (liquid_warm  ?o0 - object)
        (wet  ?o0 - object)
        (chopped  ?o0 - object)
        (clean  ?o0 - object)
        (hand_occupied  ?o0 - object)
        (visited  ?o0 - object)
    )(:functions
    (total-cost )
    (cost  ?actor - actor)
        )
        
    
    
    (:derived (indirect_on  ?o ?s) (exists ( ?o0 - object) (and (on  ?o ?o0) (or (indirect_on  ?o0 ?s) (on  ?o0 ?s)))))
    
    (:derived (reachable  ?o ?l ?a) (and (at  ?a ?l) (or (= ?l ?o) (on  ?o ?l) (indirect_on  ?o ?l) (at  ?o ?l))))
    
    (:derived (free  ?o) (not (exists ( ?os - carry) (on  ?os ?o))))
    
    (:action take_out
        :parameters ( ?actor - take_outer ?object - carry ?from - contain ?at - location)
        :precondition (and (not (hand_occupied  ?actor)) (free  ?object) (in  ?object ?from) (not (closed  ?from)) (reachable  ?from ?at ?actor))
        :effect (and (inhand  ?object ?actor) (hand_occupied  ?actor) (not (in  ?object ?from)) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action putin
        :parameters ( ?actor - putiner ?object - carry ?in - contain ?at - location)
        :precondition (and (inhand  ?object ?actor) (not (closed  ?in)) (reachable  ?in ?at ?actor))
        :effect (and (in  ?object ?in) (not (inhand  ?object ?actor)) (not (hand_occupied  ?actor)) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action chop
        :parameters ( ?actor - choper ?tool - cut ?target - consumable ?on - sturdy-support ?at - location)
        :precondition (and (inhand  ?tool ?actor) (reachable  ?target ?at ?actor) (on  ?target ?on))
        :effect (and (chopped  ?target) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action fill
        :parameters ( ?actor - filler ?source - liquid-contain ?poured_liquid - liquid ?target - liquid-contain ?at - location)
        :precondition (and (liquid_in  ?poured_liquid ?source) (inhand  ?source ?actor) (reachable  ?target ?at ?actor) (not (closed  ?source)) (not (closed  ?target)))
        :effect (and (liquid_in  ?poured_liquid ?target) (not (liquid_in  ?poured_liquid ?source)) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action heat_liquid
        :parameters ( ?actor - heat_liquider ?container - liquid-contain ?heated - heat-resistance ?liquid - liquid ?heater - heat ?on - support ?at - location)
        :precondition (and (reachable  ?heater ?at ?actor) (or (and (liquid_in  ?liquid ?container) (= ?container ?heated) (on  ?container ?on) (= ?on ?heater)) (and (liquid_in  ?liquid ?container) (= ?container ?heater)) (and (liquid_in  ?liquid ?container) (in  ?container ?heater) (= ?container ?heated))))
        :effect (and (liquid_warm  ?liquid) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action move
        :parameters ( ?actor - moveer ?start - location ?end - location)
        :precondition (and (at  ?actor ?start))
        :effect (and (not (at  ?actor ?start)) (not (at  ?start ?actor)) (at  ?actor ?end) (visited  ?start) (visited  ?end) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action wet_swipe
        :parameters ( ?actor - wet_swipeer ?tool - wet-swipe ?target - carry ?at - location)
        :precondition (and (wet  ?tool) (inhand  ?tool ?actor) (reachable  ?target ?at ?actor))
        :effect (and (clean  ?target) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action stack
        :parameters ( ?actor - stacker ?top - stack ?bottom - stack ?at - location)
        :precondition (and (reachable  ?bottom ?at ?actor) (inhand  ?top ?actor))
        :effect (and (on  ?top ?bottom) (not (inhand  ?top ?actor)) (not (hand_occupied  ?actor)) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action place
        :parameters ( ?actor - placeer ?object - carry ?on - support ?at - location)
        :precondition (and (inhand  ?object ?actor) (reachable  ?on ?at ?actor))
        :effect (and (on  ?object ?on) (not (inhand  ?object ?actor)) (not (hand_occupied  ?actor)) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action open
        :parameters ( ?actor - opener ?object - open ?at - location)
        :precondition (and (closed  ?object) (reachable  ?object ?at ?actor) (not (hand_occupied  ?actor)))
        :effect (and (not (closed  ?object)) (opened  ?object) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action handover
        :parameters ( ?actor - handoverer ?recipient - actor ?object - carry)
        :precondition (and (inhand  ?object ?actor) (at  ?actor ?recipient) (not (= ?recipient ?actor)))
        :effect (and (not (inhand  ?object ?actor)) (not (hand_occupied  ?actor)) (inhand  ?object ?recipient) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action grasp
        :parameters ( ?actor - grasper ?object - carry ?from - support ?at - location)
        :precondition (and (not (hand_occupied  ?actor)) (free  ?object) (on  ?object ?from) (reachable  ?from ?at ?actor))
        :effect (and (inhand  ?object ?actor) (hand_occupied  ?actor) (not (on  ?object ?from)) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action pour
        :parameters ( ?actor - pourer ?source - pour ?poured_object - carry ?target - contain ?at - location)
        :precondition (and (in  ?poured_object ?source) (inhand  ?source ?actor) (reachable  ?target ?at ?actor) (not (closed  ?target)))
        :effect (and (in  ?poured_object ?target) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action receive_object
        :parameters ( ?actor - receive_objecter ?giver - actor ?object - carry)
        :precondition (and (inhand  ?object ?giver) (at  ?actor ?giver) (not (hand_occupied  ?actor)))
        :effect (and (not (inhand  ?object ?giver)) (not (hand_occupied  ?giver)) (inhand  ?object ?actor) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action close
        :parameters ( ?actor - closeer ?object - close ?at - location)
        :precondition (and (opened  ?object) (reachable  ?object ?at ?actor) (not (hand_occupied  ?actor)))
        :effect (and (not (opened  ?object)) (closed  ?object) (increase (total-cost ) (cost  ?actor)))
    )
    
    (:action make-at-symmetric
        :parameters ( ?actor - actor ?other - actor)
        :precondition (and (at  ?actor ?other))
        :effect (and (at  ?other ?actor))
    )
    
    (:action dummy
        :parameters ( ?dummy - dummy ?p - pour ?o - carry ?tc - contain)
        :precondition (and (in  ?o ?p))
        :effect (and (in  ?o ?tc) (increase (total-cost ) (cost  ?dummy)))
    )
    
    (:action wetten
        :parameters ( ?ws - wet-swipe ?c - contain ?lc - liquid-contain ?wa - water)
        :precondition (and (not (wet  ?ws)) (in  ?ws ?c) (liquid_in  ?wa ?lc) (= ?c ?lc))
        :effect (and (wet  ?ws))
    )
)
