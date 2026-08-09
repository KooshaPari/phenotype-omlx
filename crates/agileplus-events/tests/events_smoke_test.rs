//! Integration smoke tests for the public event-bus contract.

use agileplus_events::{DomainEvent, EventBus};
use serde_json::json;

fn all_event_variants() -> Vec<DomainEvent> {
    vec![
        DomainEvent::FeatureCreated { id: 1 },
        DomainEvent::FeatureStateChanged {
            id: 1,
            from: "specified".into(),
            to: "implemented".into(),
        },
        DomainEvent::CycleStarted {
            cycle_id: 2,
            module_id: 3,
        },
        DomainEvent::CycleEnded { cycle_id: 2 },
        DomainEvent::WorkPackageLinked {
            work_package_id: 4,
            feature_id: 1,
        },
        DomainEvent::UserLoggedIn {
            user_id: "user-5".into(),
        },
        DomainEvent::PlaneWebhookReceived {
            issue_id: "PLANE-6".into(),
            action: "updated".into(),
        },
        DomainEvent::Custom {
            name: "integration.completed".into(),
            payload: json!({"duration_ms": 42}),
        },
    ]
}

#[test]
fn every_public_event_variant_round_trips_through_json() {
    let events = all_event_variants();
    assert_eq!(events.len(), 8, "guard: every public event variant is exercised");

    for event in events {
        let json = serde_json::to_string(&event).expect("serialize event");
        let decoded: DomainEvent = serde_json::from_str(&json).expect("deserialize event");
        assert_eq!(decoded, event);
    }
}

#[test]
fn every_public_event_variant_uses_its_stable_wire_kind() {
    let expected_kinds = [
        "feature_created",
        "feature_state_changed",
        "cycle_started",
        "cycle_ended",
        "work_package_linked",
        "user_logged_in",
        "plane_webhook_received",
        "custom",
    ];

    for (event, expected_kind) in all_event_variants().into_iter().zip(expected_kinds) {
        let value = serde_json::to_value(event).expect("serialize event");
        assert_eq!(value["kind"], expected_kind);
    }
}

#[tokio::test]
async fn event_bus_delivers_each_public_event_variant() {
    let bus = EventBus::new(8);
    let mut subscriber = bus.subscribe();

    for event in all_event_variants() {
        assert_eq!(bus.publish(event.clone()).expect("publish event"), 1);
        assert_eq!(subscriber.recv().await.expect("receive event"), event);
    }
}

#[tokio::test]
async fn event_bus_fans_out_to_each_subscriber() {
    let bus = EventBus::new(4);
    let mut first = bus.subscribe();
    let mut second = bus.subscribe();
    let event = DomainEvent::FeatureCreated { id: 42 };

    assert_eq!(bus.publish(event.clone()).expect("publish event"), 2);
    assert_eq!(first.recv().await.expect("first subscriber event"), event);
    assert_eq!(second.recv().await.expect("second subscriber event"), event);
}

#[tokio::test]
async fn event_bus_async_publish_preserves_payload() {
    let bus = EventBus::new(1);
    let mut subscriber = bus.subscribe();
    let event = DomainEvent::PlaneWebhookReceived {
        issue_id: "issue-42".into(),
        action: "created".into(),
    };

    assert_eq!(
        bus.publish_async(event.clone())
            .await
            .expect("publish event"),
        1,
        "async publish reports the sole subscriber"
    );
    assert_eq!(subscriber.recv().await.expect("subscriber event"), event);
}
