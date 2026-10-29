"""Native visit identity, physical clocks and observed service opportunities."""
import json

from experiments.mcwm_closed_loop_service_opportunity_v77_20261027.core import compare_services


def visit(bus, station, arrival, departure, *, aq=(), ao=(), dq=(), do=(), ready=True):
    def row(kind, clock, queue, onboard):
        return dict(record='service',kind=kind,bus_id=bus,line_id='L',station_id=station,
            clock_seconds=clock,observed_at=clock,observation_order=0,
            queued_passenger_ids=list(queue),onboard_passenger_ids=list(onboard),
            capacity=50,free_capacity=50-len(onboard),eligible_waiting=len(queue),
            native_board_count=None,native_alight_count=None,native_stranded_count=None)
    values=[row('arrival',arrival,aq,ao)]
    if ready:values.append(row('service_ready',(arrival+departure)/2,aq,ao))
    values.append(row('departure',departure,dq,do))
    return values


def ordered(*groups):
    rows=[row for group in groups for row in group]
    for i,row in enumerate(rows):row['observation_order']=i
    return rows


def issue(trip, station, clock, hold):
    return dict(trip_id=trip,vehicle_id='shared_fleet_0',line_id='L',station_id=station,
                started_at=clock,a=[hold/30-1])


def test_trip_identity_prevents_reused_fleet_from_shifting_holding_pairs():
    f=ordered(visit('t0','A',12010,12030),visit('t1','A',12050,12070))
    b=ordered(visit('t0','A',12010,12030),visit('t1','A',12050,12070))
    result=compare_services(f,b,[issue('t0','A',12020,1),issue('t1','A',12060,2)],
        [issue('t1','A',12060,4)])
    delta=result['all']['timing_delta']['holding']
    assert delta['count']==1 and abs(delta['sum']+2)<1e-12
    assert result['alignment']['issue_alignment']['fresh']['unique_issue_visits']==2
    assert result['alignment']['issue_alignment']['baseline']['visits_without_issue']==1


def test_half_open_window_preserves_crossings_and_one_arm_only_visits():
    f=ordered(visit('lo','A',11995,12005),visit('hi','A',16195,16205),
        visit('after','A',16200,16210),visit('before','A',11980,12000),visit('one','A',12010,12020))
    b=ordered(visit('lo','A',11996,12006),visit('hi','A',16195,16205),
        visit('after','A',16200,16210),visit('before','A',11980,12000),visit('one','A',11980,11990))
    result=compare_services(f,b,[],[])
    assert result['alignment']['selected_common_visits']==3
    bounds=result['alignment']['boundaries']
    assert bounds['crosses_lower_boundary']==bounds['crosses_upper_boundary']==bounds['only_one_arm_overlaps']==1
    assert bounds['internal_both_arms']==0
    assert result['all']['timing_delta']['holding']['sum'] is None


def test_lagged_physical_endpoints_and_null_legacy_values_are_not_zero_filled():
    f=ordered(visit('t','A',12010,12030,aq=['p'],do=['p']))
    b=ordered(visit('t','A',12010,12030,aq=['p'],do=['p']))
    f[0]['observed_at']+=.5
    f[1]['observed_at']+=.5
    b[1]['clock_seconds']-=1
    b[1]['observed_at']=b[1]['clock_seconds']+.5
    f[-1]['free_capacity']=None
    result=compare_services(f,b,[issue('t','A',12020,1)],[issue('t','A',12020,1)])
    delta=result['all']['timing_delta']
    assert delta['arrival']['count']==delta['dwell']['count']==delta['holding']['count']==0
    assert delta['ready']['sum']==1
    assert result['alignment']['ready']['fresh']['lagged_unique']==1
    fields=result['all']['measurements']['departure']
    assert fields['native_stranded_count']['fresh']['missing']==1
    assert fields['native_stranded_count']['fresh']['sum'] is None
    assert fields['free_capacity']['paired_delta']['sum'] is None
    assert result['all']['inventory_changes']['paired_delta']['observed_visits']==0
    assert result['direct_opportunities']['coverage']['missing_or_lagged_arrival_queue']==1


def test_board_versus_wait_is_directional_and_deduplicated_by_person_stop():
    fresh=[];baseline=[]
    for bus,station,start in [('a','A',12010),('b','A',12110),('c','B',12210)]:
        fresh.extend(visit(bus,station,start,start+20,aq=['p','q'],dq=['q'],do=['p']))
        baseline.extend(visit(bus,station,start,start+20,aq=['p','q'],dq=['p'],do=['q']))
    result=compare_services(ordered(fresh),ordered(baseline),[],[])
    for direction in result['direct_opportunities']['directions'].values():
        assert direction['observed_events']==3
        assert direction['person_stop_event_memberships']==3
        assert direction['unique_person_stop_pairs']==2
        assert direction['first_witness_events']==2
        assert direction['examples'][0]['bus']=='a'
        assert all(row['legacy_compatibility_available'] is False for row in direction['examples'])
    encoded=json.dumps(result)
    assert '"p"' not in encoded and '"q"' not in encoded


def test_onboard_membership_changes_and_capacity_risk_are_observed_once_per_visit():
    f=ordered(visit('t','A',12010,12030,ao=['old'],do=['new'],dq=['queue']))
    b=ordered(visit('t','A',12010,12030,ao=['old'],do=['old'],dq=['queue']))
    f[-1]['free_capacity']=0
    f[-1]['native_stranded_count']=3
    b[-1]['native_stranded_count']=None
    result=compare_services(f,b,[],[])
    inv=result['all']['inventory_changes']['paired_delta']
    assert inv['added']['sum']==inv['removed']['sum']==1
    assert result['all']['departure_risk']['paired']['fresh_only_risk']==1
    legacy=result['all']['measurements']['departure']['native_stranded_count']
    assert legacy['fresh']['sum']==3 and legacy['baseline']['sum'] is None
    assert legacy['paired_delta']['sum'] is None


def test_service_order_counts_strict_inversion_and_keeps_ties_separate():
    f=ordered(visit('a','A',12030,12040),visit('b','A',12015,12025),visit('c','A',12035,12045))
    b=ordered(visit('a','A',12010,12020),visit('b','A',12020,12030),visit('c','A',12020,12030))
    result=compare_services(f,b,[],[])
    counts=result['service_order']['counts']
    assert counts['strict_adjacent_inversions']==1 and counts['tied_pairs']==1
    assert result['service_order']['examples'][0]['baseline_bus_order']==['a','b']
