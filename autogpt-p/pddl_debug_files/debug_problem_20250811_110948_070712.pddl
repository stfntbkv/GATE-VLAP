(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        trash_can0 - trash_can
        pepsi0 - pepsi
        counter1 - counter
        multigrain_chips0 - multigrain_chips
        sponge0 - sponge
        red_bull0 - red_bull
        7up0 - 7up
        water0 - water
        robot0 - robot_profile
        table0 - table
        human0 - human
        energy_bar0 - energy_bar
        grapefruit_soda0 - grapefruit_soda
        tea0 - tea
        coke0 - coke
        apple0 - apple
        jalapeno_chips0 - jalapeno_chips
        sprite0 - sprite
        counter2 - counter
        lime_soda0 - lime_soda
    )
    
    (:init 
        (on  jalapeno_chips0 counter2)
        (on  energy_bar0 counter1)
        (on  7up0 counter2)
        (on  red_bull0 table0)
        (on  multigrain_chips0 counter2)
        (on  lime_soda0 counter2)
        (on  grapefruit_soda0 counter1)
        (on  coke0 counter1)
        (on  tea0 counter2)
        (on  sponge0 counter1)
        (on  water0 counter2)
        (on  pepsi0 table0)
        (on  sprite0 table0)
        (on  apple0 counter2)
        (at  robot0 counter1)
        (= total-cost 0)
        (= (cost robot0) 1)
        (= (cost human0) 100)
    )
    
    (:goal (and (inhand  lime_soda0 robot0) (inhand  multigrain_chips0 robot0)))
    (:metric minimize (total-cost))
    
)
