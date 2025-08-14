(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        jalapeno_chips0 - jalapeno_chips
        multigrain_chips0 - multigrain_chips
        tea0 - tea
        7up0 - 7up
        lime_soda0 - lime_soda
        apple0 - apple
        red_bull0 - red_bull
        energy_bar0 - energy_bar
        robot0 - robot_profile
        counter1 - counter
        sprite0 - sprite
        counter2 - counter
        trash_can0 - trash_can
        sponge0 - sponge
        pepsi0 - pepsi
        human0 - human
        coke0 - coke
        grapefruit_soda0 - grapefruit_soda
        table0 - table
        water0 - water
    )
    
    (:init 
        (on  sponge0 counter1)
        (on  energy_bar0 counter1)
        (on  sprite0 table0)
        (on  apple0 counter2)
        (on  water0 counter2)
        (on  pepsi0 table0)
        (on  7up0 counter2)
        (on  red_bull0 table0)
        (on  multigrain_chips0 counter2)
        (on  coke0 counter1)
        (on  tea0 counter2)
        (on  jalapeno_chips0 counter2)
        (at  robot0 counter1)
        (on  grapefruit_soda0 counter1)
        (on  lime_soda0 counter2)
        (= total-cost 0)
        (= (cost robot0) 1)
        (= (cost human0) 100)
    )
    
    (:goal (and (inhand  7up0 human0) (inhand  tea0 human0)))
    (:metric minimize (total-cost))
    
)
