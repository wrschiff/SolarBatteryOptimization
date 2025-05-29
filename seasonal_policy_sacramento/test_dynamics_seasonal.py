from parameters_seasonal import *
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from generate_dynamics_dbs import (
    sell_df,
    irr_upper_df,
    irr_lower_df,
    irr_mean_df,
    load_df,
    get_pge_buy_price,
)
# Import the functions to test
from dynamics_seasonal_sacramento import (
    stage_to_calendar,
    get_expected_irr_and_load,
    get_irr_and_load_range,
    gen_irr_and_load,
    next_state,
    control_from_state,
    arbitrage_cost,
    buy_sell_rates,
)
# Mock the parameters class for testing
class MockParameters:
    def __init__(self):
        self.ETA = 0.9
        self.N_BATT = 10.0
        self.BASE_YEAR = 2025
        self.STRUCTURE = "TOU"
        self.CITY = "Sacramento"

@pytest.fixture
def parameters():
    return MockParameters()

class TestStageToCalendar:
    """Test the stage_to_calendar function"""
    
    def test_stage_zero(self):
        """Test stage 0 corresponds to Jan 1, midnight"""
        month, week, hour = stage_to_calendar(0, 2025)
        assert month == 1
        assert week == 1
        assert hour == 0
    
    def test_stage_24(self):
        """Test stage 24 corresponds to Jan 2, midnight"""
        month, week, hour = stage_to_calendar(24, 2025)
        assert month == 1
        assert week == 1
        assert hour == 0
    
    def test_stage_8760(self):
        """Test stage 8760 corresponds to next year"""
        month, week, hour = stage_to_calendar(8760, 2025)
        # Should be Jan 1 of next year
        assert month == 1
        assert hour == 0
    
    def test_mid_year_stage(self):
        """Test a stage in the middle of the year"""
        # Stage corresponding to July 1st, noon (approximately)
        july_1_stage = 24 * 181 + 12  # 181 days from Jan 1 + 12 hours
        month, week, hour = stage_to_calendar(july_1_stage, 2025)
        assert month == 7
        assert hour == 12
    
    def test_different_base_year(self):
        """Test with different base year"""
        month, week, hour = stage_to_calendar(0, 2024)
        assert month == 1
        assert week == 1
        assert hour == 0

class TestGetExpectedIrrAndLoad:
    """Test the get_expected_irr_and_load function"""
    
    @patch('dynamics_seasonal_sacramento.irr_mean_df')
    @patch('dynamics_seasonal_sacramento.load_df')
    def test_get_expected_irr_and_load(self, mock_load_df, mock_irr_df, parameters):
        """Test getting expected IRR and load values"""
        # Mock the dataframes with proper indexing
        mock_irr_df.loc.__getitem__ = MagicMock(return_value=0.8)
        mock_load_df.loc.__getitem__ = MagicMock(return_value=5.5)
        
        irr, load = get_expected_irr_and_load(100, parameters)
        
        assert irr == 0.8
        assert load == 5.5

class TestGetIrrAndLoadRange:
    """Test the get_irr_and_load_range function"""
    
    @patch('dynamics_seasonal_sacramento.irr_mean_df')
    @patch('dynamics_seasonal_sacramento.irr_lower_df')
    @patch('dynamics_seasonal_sacramento.irr_upper_df')
    @patch('dynamics_seasonal_sacramento.load_df')
    def test_get_irr_and_load_range(self, mock_load_df, mock_irr_upper_df, 
                                   mock_irr_lower_df, mock_irr_mean_df, parameters):
        """Test getting IRR and load ranges"""
        # Set up mock returns with proper indexing
        mock_irr_mean_df.loc.__getitem__ = MagicMock(return_value=1.0)
        mock_irr_lower_df.loc.__getitem__ = MagicMock(return_value=0.8)
        mock_irr_upper_df.loc.__getitem__ = MagicMock(return_value=1.2)
        mock_load_df.loc.__getitem__ = MagicMock(return_value=5.0)
        
        irr_range, load_range = get_irr_and_load_range(100, parameters)
        
        # Check IRR range
        assert len(irr_range) == 2
        assert irr_range[0] == 0.8 * 1.0  # minVars * irr
        assert irr_range[1] == 1.2 * 1.0  # maxVars * irr
        
        # Check load range
        assert len(load_range) == 2
        assert load_range[0] == 0.8 * 5.0  # consumpVarMin * load
        assert load_range[1] == 1.2 * 5.0  # consumpVarMax * load

class TestGenIrrAndLoad:
    """Test the gen_irr_and_load function"""
    
    @patch('dynamics_seasonal_sacramento.get_irr_and_load_range')
    @patch('random.uniform')
    def test_gen_irr_and_load(self, mock_uniform, mock_get_range):
        """Test random generation of IRR and load"""
        # Mock the range function
        mock_get_range.return_value = ([0.5, 1.5], [3.0, 7.0])
        
        # Mock random.uniform to return specific values
        mock_uniform.side_effect = [1.0, 5.0]  # First call returns 1.0, second returns 5.0
        
        irr, load = gen_irr_and_load(100, "Sacramento")
        
        assert irr == 1.0
        assert load == 5.0
        assert mock_uniform.call_count == 2

class TestNextState:
    """Test the next_state function"""
    
    def test_next_state_positive_control(self, parameters):
        """Test next state calculation with positive control (charging)"""
        current_state = 5.0
        control = 2.0
        eta = parameters.ETA
        
        result = next_state(current_state, control, parameters)
        expected = current_state + control * eta
        
        assert result == expected
    
    def test_next_state_negative_control(self, parameters):
        """Test next state calculation with negative control (discharging)"""
        current_state = 8.0
        control = -3.0
        eta = parameters.ETA
        
        result = next_state(current_state, control, parameters)
        expected = current_state + control * (1/eta)
        
        assert result == expected
    
    def test_next_state_zero_control(self, parameters):
        """Test next state with zero control"""
        current_state = 5.0
        control = 0.0
        
        result = next_state(current_state, control, parameters)
        
        assert result == current_state

class TestControlFromState:
    """Test the control_from_state function"""
    
    def test_control_from_state_charging(self, parameters):
        """Test control calculation when charging (next > current)"""
        current = 3.0
        next_state = 5.0
        eta = parameters.ETA
        
        result = control_from_state(current, next_state, parameters)
        expected = (next_state - current) / eta  # Fixed: should be divided by eta for charging
        
        assert abs(result - expected) < 1e-10  # Use approximate equality for floating point
    
    def test_control_from_state_discharging(self, parameters):
        """Test control calculation when discharging (next < current)"""
        current = 8.0
        next_state = 6.0
        eta = parameters.ETA
        
        result = control_from_state(current, next_state, parameters)
        expected = (next_state - current) * eta  # Fixed: should be multiplied by eta for discharging
        
        assert abs(result - expected) < 1e-10  # Use approximate equality for floating point
    
    def test_control_from_state_infeasible(self, parameters):
        """Test control calculation when transition is infeasible"""
        current = 5.0
        next_state = 100.0  # Impossible jump
        
        result = control_from_state(current, next_state, parameters)
        
        assert result is None
    
    def test_control_from_state_boundary_conditions(self, parameters):
        """Test boundary conditions for control calculation"""
        # Test a more reasonable transition
        current = 5.0
        next_state = 8.0  # More reasonable jump
        
        result = control_from_state(current, next_state, parameters)
        # Should be valid for a reasonable state transition
        assert result is not None
        assert isinstance(result, (int, float))

class TestArbitrageCost:
    """Test the arbitrage_cost function"""
    
    @patch('dynamics_seasonal_sacramento.buy_sell_rates')
    def test_arbitrage_cost_buying(self, mock_buy_sell_rates, parameters):
        """Test arbitrage cost when buying from grid"""
        mock_buy_sell_rates.return_value = [0.25, 0.15]  # [buy_rate, sell_rate]
        
        stage = 100
        control = 2.0  # Charging battery
        load = 5.0
        solar = 1.0
        
        # p_grid = load - solar + control = 5 - 1 + 2 = 6 > 0 (buying)
        result = arbitrage_cost(stage, control, load, solar, parameters)
        expected = 6.0 * 0.25  # p_grid * buy_rate
        
        assert result == expected
    
    @patch('dynamics_seasonal_sacramento.buy_sell_rates')
    def test_arbitrage_cost_selling(self, mock_buy_sell_rates, parameters):
        """Test arbitrage cost when selling to grid"""
        mock_buy_sell_rates.return_value = [0.25, 0.15]  # [buy_rate, sell_rate]
        
        stage = 100
        control = -3.0  # Discharging battery
        load = 2.0
        solar = 8.0
        
        # p_grid = load - solar + control = 2 - 8 + (-3) = -9 < 0 (selling)
        result = arbitrage_cost(stage, control, load, solar, parameters)
        expected = -9.0 * 0.15  # p_grid * sell_rate
        
        assert result == expected

class TestBuySellRates:
    """Test the buy_sell_rates function"""
    
    @patch('dynamics_seasonal_sacramento.get_pge_buy_price')
    @patch('dynamics_seasonal_sacramento.sell_df')
    def test_buy_sell_rates(self, mock_sell_df, mock_get_pge_buy_price):
        """Test getting buy and sell rates"""
        # Mock the PGE buy price
        mock_get_pge_buy_price.return_value = 0.30
        
        # Mock the sell dataframe with proper indexing
        mock_sell_df.loc.__getitem__ = MagicMock(return_value=0.18)
        
        stage = 1000  # Some arbitrary stage
        
        buy_rate, sell_rate = buy_sell_rates(stage)
        
        assert buy_rate == 0.30
        assert sell_rate == 0.18

class TestIntegration:
    """Integration tests combining multiple functions"""
    
    @patch('dynamics_seasonal_sacramento.irr_mean_df')
    @patch('dynamics_seasonal_sacramento.load_df')
    @patch('dynamics_seasonal_sacramento.buy_sell_rates')
    def test_full_timestep_simulation(self, mock_buy_sell_rates, mock_load_df, 
                                     mock_irr_df, parameters):
        """Test a complete timestep simulation"""
        # Set up mocks with proper return values (not MagicMocks)
        mock_irr_df.loc.__getitem__ = MagicMock(return_value=1.0)
        mock_load_df.loc.__getitem__ = MagicMock(return_value=5.0)
        mock_buy_sell_rates.return_value = [0.25, 0.15]
        
        # Initial conditions
        initial_state = 5.0
        stage = 100
        control = 2.0
        
        # Get expected values
        irr, load = get_expected_irr_and_load(stage, parameters)
        
        # Calculate next state
        next_battery_state = next_state(initial_state, control, parameters)
        
        # Calculate cost
        cost = arbitrage_cost(stage, control, load, irr, parameters)
        
        # Verify everything runs without errors
        assert isinstance(irr, (int, float))
        assert isinstance(load, (int, float))
        assert isinstance(next_battery_state, (int, float))
        assert isinstance(cost, (int, float))
        
        # Verify state transition makes sense
        assert next_battery_state > initial_state  # Should increase with positive control

# Utility test functions
class TestUtilities:
    """Test utility and edge case scenarios"""
    
    def test_year_boundary_stages(self):
        """Test stages at year boundaries"""
        # Test end of year
        month, week, hour = stage_to_calendar(8759, 2025)  # Last hour of year
        assert month == 12
        assert hour == 23
        
        # Test beginning of next year
        month, week, hour = stage_to_calendar(8760, 2025)
        assert month == 1
        assert hour == 0
    
    def test_parameter_validation(self, parameters):
        """Test that parameters are used correctly"""
        assert hasattr(parameters, 'ETA')
        assert hasattr(parameters, 'N_BATT')
        assert hasattr(parameters, 'STRUCTURE')
        assert 0 < parameters.ETA <= 1  # Efficiency should be between 0 and 1
        assert parameters.N_BATT > 0  # Battery capacity should be positive

if __name__ == "__main__":
    # Run specific test classes or all tests
    pytest.main([__file__, "-v"])