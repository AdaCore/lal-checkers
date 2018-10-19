procedure Test is
   X : Integer;
begin
   X := 3;
exception
   when others =>
      X := 2;
end Test;
