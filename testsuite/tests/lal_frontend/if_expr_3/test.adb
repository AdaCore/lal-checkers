procedure Test is
   x : Integer;
   C1, C2 : Boolean;
   A1, A2 : Integer;
begin
   x := if (if C1 then C1 else C2) then A1 else A2
end Ex1;
