(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        trash_can0 - trash_can
        lime_soda0 - lime_soda
        multigrain_chips0 - multigrain_chips
        table0 - table
        apple0 - apple
        7up0 - 7up
        coke0 - coke
        counter1 - counter
        counter2 - counter
        sponge0 - sponge
        tea0 - tea
        human0 - human
        red_bull0 - red_bull
        pepsi0 - pepsi
        water0 - water
        energy_bar0 - energy_bar
        jalapeno_chips0 - jalapeno_chips
        grapefruit_soda0 - grapefruit_soda
        robot0 - robot_profile
        sprite0 - sprite
    )
    
    (:init 
        (on  sprite0 table0)
        (on  multigrain_chips0 counter2)
        (on  jalapeno_chips0 counter2)
        (on  coke0 counter1)
        (on  sponge0 counter1)
        (on  energy_bar0 counter1)
        (on  red_bull0 table0)
        (on  apple0 counter2)
        (on  pepsi0 table0)
        (on  grapefruit_soda0 counter1)
        (at  robot0 counter1)
        (on  tea0 counter2)
        (on  7up0 counter2)
        (on  lime_soda0 counter2)
        (on  water0 counter2)
        (= total-cost 0)
        (= (cost human0) 100)
        (= (cost robot0) 1)
    )
    
    (:goal (or (and (on  multigrain_chips0 table0) (on  apple0 counter2))))
    (:metric minimize (total-cost))
    
)
