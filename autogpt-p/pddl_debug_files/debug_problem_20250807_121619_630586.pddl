(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        table0 - table
        lime_soda0 - lime_soda
        trash_can0 - trash_can
        human0 - human
        water0 - water
        multigrain_chips0 - multigrain_chips
        sprite0 - sprite
        pepsi0 - pepsi
        counter2 - counter
        coke0 - coke
        7up0 - 7up
        robot0 - robot_profile
        tea0 - tea
        counter1 - counter
        sponge0 - sponge
        jalapeno_chips0 - jalapeno_chips
        red_bull0 - red_bull
        grapefruit_soda0 - grapefruit_soda
        energy_bar0 - energy_bar
        apple0 - apple
    )
    
    (:init 
        (on  7up0 counter2)
        (on  pepsi0 table0)
        (on  apple0 counter2)
        (at  robot0 counter1)
        (on  water0 counter2)
        (on  multigrain_chips0 counter2)
        (on  grapefruit_soda0 counter1)
        (on  tea0 counter2)
        (on  sprite0 table0)
        (on  jalapeno_chips0 counter2)
        (on  coke0 counter1)
        (on  energy_bar0 counter1)
        (on  sponge0 counter1)
        (on  red_bull0 table0)
        (on  lime_soda0 counter2)
        (= total-cost 0)
        (= (cost human0) 100)
        (= (cost robot0) 1)
    )
    
    (:goal (and (on  energy_bar0 table0) (on  water0 table0)))
    (:metric minimize (total-cost))
    
)
