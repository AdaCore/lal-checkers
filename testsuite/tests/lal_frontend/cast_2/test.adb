procedure Test is
    procedure Foo (I : out Integer) is
    begin
       I := -2;
    end Foo;

    X : Natural := 0;
begin
    Foo (Integer (X));
end Test;