(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        jalapeno_chips0 - jalapeno_chips
        coke0 - coke
        apple0 - apple
        table0 - table
        lime_soda0 - lime_soda
        sponge0 - sponge
        water0 - water
        red_bull0 - red_bull
        7up0 - 7up
        counter2 - counter
        energy_bar0 - energy_bar
        human0 - human
        trash_can0 - trash_can
        tea0 - tea
        counter1 - counter
        pepsi0 - pepsi
        multigrain_chips0 - multigrain_chips
        grapefruit_soda0 - grapefruit_soda
        robot0 - robot_profile
        sprite0 - sprite
    )
    
    (:init 
        (on  red_bull0 table0)
        (on  multigrain_chips0 counter2)
        (on  tea0 counter2)
        (on  apple0 counter2)
        (on  jalapeno_chips0 counter2)
        (on  water0 counter2)
        (on  energy_bar0 counter1)
        (on  7up0 counter2)
        (on  pepsi0 table0)
        (at  robot0 counter1)
        (on  grapefruit_soda0 counter1)
        (on  lime_soda0 counter2)
        (on  coke0 counter1)
        (on  sprite0 table0)
        (on  sponge0 counter1)
        (= total-cost 0)
        (= (cost human0) 100)
        (= (cost robot0) 1)
    )
    
    (:goal (and (on  energy_bar0 table0) (on  water0 table0)))
    (:metric minimize (total-cost))
    
)
