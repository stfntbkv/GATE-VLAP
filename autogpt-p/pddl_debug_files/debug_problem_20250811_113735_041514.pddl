(define (problem test)

    (:domain robotic_planning)
    
    (:objects
        trash_can0 - trash_can
        sponge0 - sponge
        tea0 - tea
        water0 - water
        lime_soda0 - lime_soda
        energy_bar0 - energy_bar
        multigrain_chips0 - multigrain_chips
        human0 - human
        jalapeno_chips0 - jalapeno_chips
        table0 - table
        red_bull0 - red_bull
        grapefruit_soda0 - grapefruit_soda
        robot0 - robot_profile
        counter1 - counter
        apple0 - apple
        counter2 - counter
    )
    
    (:init 
        (on  multigrain_chips0 counter2)
        (on  jalapeno_chips0 counter2)
        (on  sponge0 counter1)
        (on  energy_bar0 counter1)
        (on  red_bull0 table0)
        (on  apple0 counter2)
        (on  grapefruit_soda0 counter1)
        (at  robot0 counter1)
        (on  tea0 counter2)
        (on  lime_soda0 counter2)
        (on  water0 counter2)
        (= total-cost 0)
        (= (cost human0) 100)
        (= (cost robot0) 1)
    )
    
    (:goal (or (and (on  grapefruit_soda0 counter1) (on  lime_soda0 counter1)) (and (on  grapefruit_soda0 counter1) (on  grapefruit_soda0 counter1))))
    (:metric minimize (total-cost))
    
)
